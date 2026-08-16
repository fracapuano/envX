from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dm_control import suite
from dm_control.suite import reacher as dmc_reacher
from mujoco import mjx

from envx.reacher import create_reacher
from envx.reacher.env import _render_batch


def _physics_env(
    num_worlds: int = 4,
    episode_length: int = 1000,
    task: str = "easy",
    *,
    visualize_goal: bool = False,
):
    return create_reacher(
        num_worlds,
        (16, 16),
        task=task,
        observation_type="states",
        episode_length=episode_length,
        physics_backend="jax",
        render=False,
        visualize_goal=visualize_goal,
    )


def _native_env(task: str = "easy"):
    return suite.load("reacher", task, task_kwargs={"random": 0})


def _warp_cuda_available() -> bool:
    try:
        from mujoco.mjx.warp import io as warp_io

        return bool(jax.local_devices(backend="gpu")) and bool(warp_io.has_cuda_gpu_device())
    except (ImportError, RuntimeError):
        return False


def _warp_available() -> bool:
    try:
        import mujoco.mjx.warp as mjx_warp

        return bool(mjx_warp.WARP_INSTALLED)
    except ImportError:
        return False


@pytest.mark.parametrize(("task", "target_size"), [("easy", 0.05), ("hard", 0.015)])
def test_model_is_derived_from_dmc_reacher(task, target_size):
    env = _physics_env(1, task=task)
    model = env.mj_model
    reference = dmc_reacher.Physics.from_xml_string(*dmc_reacher.get_model_and_assets()).model.ptr

    assert model.nq == reference.nq == 2
    assert model.nv == reference.nv == 2
    assert model.nu == reference.nu == 2
    assert model.nmocap == 1  # Per-world form of DMC's static target geom.
    assert model.opt.integrator == reference.opt.integrator
    assert model.opt.timestep == reference.opt.timestep == pytest.approx(0.02)
    np.testing.assert_array_equal(model.actuator_ctrlrange, reference.actuator_ctrlrange)
    np.testing.assert_array_equal(model.actuator_gear, reference.actuator_gear)
    np.testing.assert_array_equal(model.jnt_range, reference.jnt_range)
    np.testing.assert_array_equal(model.dof_damping, reference.dof_damping)
    np.testing.assert_array_equal(model.dof_armature, reference.dof_armature)

    for name in ("root", "arm", "hand", "finger"):
        np.testing.assert_array_equal(model.geom(name).size, reference.geom(name).size)
        np.testing.assert_array_equal(model.geom(name).rgba, reference.geom(name).rgba)
    assert model.geom("target").size[0] == pytest.approx(target_size)
    assert model.nmat == reference.nmat
    assert model.ntex == reference.ntex

    camera_id = model.camera("fixed").id
    reference_camera_id = reference.camera("fixed").id
    np.testing.assert_array_equal(model.cam_pos[camera_id], reference.cam_pos[reference_camera_id])
    np.testing.assert_array_equal(
        model.cam_quat[camera_id], reference.cam_quat[reference_camera_id]
    )
    np.testing.assert_array_equal(
        model.cam_fovy[camera_id], reference.cam_fovy[reference_camera_id]
    )

    assert env.frame_skip == 1
    assert env.dt == pytest.approx(0.02)
    assert env.episode_length == 1000
    assert env.time_limit == pytest.approx(20.0)
    assert env.observation_shape == (6,)
    pixel_env = create_reacher(1, physics_backend="warp", render=False, observation_type="pixels")
    assert pixel_env.observation_shape == (240, 320, 3)


def test_reset_is_deterministic_and_follows_dmc_distributions():
    env = _physics_env(4096)
    keys = jax.random.split(jax.random.key(17), env.num_worlds)
    state_a = env.reset(keys)
    state_b = env.reset(keys)
    qpos = np.asarray(state_a.data.qpos)
    qvel = np.asarray(state_a.data.qvel)
    targets = np.asarray(state_a.data.mocap_pos[:, env._target_mocap_id])

    np.testing.assert_array_equal(state_a.data.qpos, state_b.data.qpos)
    np.testing.assert_array_equal(state_a.data.qvel, state_b.data.qvel)
    np.testing.assert_array_equal(state_a.data.mocap_pos, state_b.data.mocap_pos)
    assert np.all(qpos[:, 0] >= -np.pi)
    assert np.all(qpos[:, 0] < np.pi)
    wrist_range = env.mj_model.joint("wrist").range
    assert np.all(qpos[:, 1] >= wrist_range[0])
    assert np.all(qpos[:, 1] < wrist_range[1])
    np.testing.assert_array_equal(qvel, 0.0)

    target_radius = np.linalg.norm(targets[:, :2], axis=-1)
    assert np.all(target_radius >= 0.05)
    assert np.all(target_radius < 0.20)
    np.testing.assert_allclose(targets[:, 2], 0.01)
    np.testing.assert_allclose(targets[:, :2].mean(axis=0), 0.0, atol=0.004)
    assert target_radius.mean() == pytest.approx(0.125, abs=0.003)


def test_state_observation_matches_dmc_field_order():
    env = _physics_env(8)
    state = env.reset(jax.random.split(jax.random.key(3), env.num_worlds))
    data = state.data
    expected = np.concatenate(
        (
            np.asarray(data.qpos),
            np.asarray(
                data.geom_xpos[:, env._target_geom_id, :2]
                - data.geom_xpos[:, env._finger_geom_id, :2]
            ),
            np.asarray(data.qvel),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(state.obs, expected, atol=1e-7)


@pytest.mark.parametrize("task", ["easy", "hard"])
def test_mjx_transition_observation_and_reward_match_native_dmc(task):
    env = _physics_env(3, task=task)
    state = env.reset(jax.random.split(jax.random.key(9), env.num_worlds))
    actions = np.array([[0.1, -0.2], [-0.3, 0.4], [1.0, -1.0]], np.float32)
    actual = env.step(state, jnp.asarray(actions))

    for index in range(env.num_worlds):
        reference = _native_env(task)
        reference.reset()
        target = np.asarray(state.data.mocap_pos[index, env._target_mocap_id], dtype=np.float64)
        with reference.physics.reset_context():
            reference.physics.data.qpos[:] = np.asarray(state.data.qpos[index], dtype=np.float64)
            reference.physics.data.qvel[:] = np.asarray(state.data.qvel[index], dtype=np.float64)
            reference.physics.named.model.geom_pos["target"][:] = target
        timestep = reference.step(actions[index].astype(np.float64))
        expected_obs = np.concatenate(tuple(timestep.observation.values()))

        np.testing.assert_allclose(actual.data.qpos[index], reference.physics.data.qpos, atol=3e-7)
        np.testing.assert_allclose(actual.data.qvel[index], reference.physics.data.qvel, atol=3e-6)
        np.testing.assert_allclose(actual.obs[index], expected_obs, atol=3e-6)
        assert float(actual.reward[index]) == pytest.approx(timestep.reward, abs=1e-7)
        assert float(actual.discount[index]) == pytest.approx(timestep.discount)
        assert bool(actual.success[index]) == bool(timestep.reward)


def test_easy_and_hard_use_dmc_success_radii():
    easy = _physics_env(1, task="easy")
    hard = _physics_env(1, task="hard")
    assert easy._success_radius == pytest.approx(0.06)
    assert hard._success_radius == pytest.approx(0.025)


def test_goal_visualization_is_opt_in_and_dynamics_are_invariant():
    hidden = _physics_env(3)
    visible = _physics_env(3, visualize_goal=True)

    assert hidden.visualize_goal is False
    assert visible.visualize_goal is True
    assert hidden.mj_model.geom("target").group == 5
    assert visible.mj_model.geom("target").group == 0

    keys = jax.random.split(jax.random.key(41), hidden.num_worlds)
    actions = jnp.asarray([[0.2, -0.1], [-0.4, 0.7], [1.0, -1.0]])
    hidden_state = hidden.step(hidden.reset(keys), actions)
    visible_state = visible.step(visible.reset(keys), actions)
    jax.block_until_ready((hidden_state, visible_state))

    np.testing.assert_array_equal(hidden_state.data.qpos, visible_state.data.qpos)
    np.testing.assert_array_equal(hidden_state.data.qvel, visible_state.data.qvel)
    np.testing.assert_array_equal(hidden_state.data.mocap_pos, visible_state.data.mocap_pos)
    np.testing.assert_array_equal(hidden_state.obs, visible_state.obs)
    np.testing.assert_array_equal(hidden_state.reward, visible_state.reward)
    np.testing.assert_array_equal(hidden_state.distance, visible_state.distance)
    np.testing.assert_array_equal(hidden_state.success, visible_state.success)


@pytest.mark.skipif(not _warp_available(), reason="MJX-Warp is not installed.")
def test_warp_physics_matches_mjx_jax_backend():
    keys = jax.random.split(jax.random.key(22), 3)
    actions = jnp.array([[0.1, -0.2], [-0.3, 0.4], [1.0, -1.0]])
    jax_env = create_reacher(
        3, (8, 8), observation_type="states", physics_backend="jax", render=False
    )
    warp_env = create_reacher(
        3,
        (8, 8),
        observation_type="states",
        physics_backend="warp",
        render=False,
    )

    jax_state = jax_env.step(jax_env.reset(keys), actions)
    warp_state = warp_env.step(warp_env.reset(keys), actions)
    jax.block_until_ready((jax_state, warp_state))

    np.testing.assert_allclose(warp_state.data.qpos, jax_state.data.qpos, atol=1e-7)
    np.testing.assert_allclose(warp_state.data.qvel, jax_state.data.qvel, atol=3e-6)
    np.testing.assert_allclose(warp_state.obs, jax_state.obs, atol=3e-6)
    np.testing.assert_allclose(warp_state.reward, jax_state.reward, atol=1e-7)


@pytest.mark.skipif(not _warp_available(), reason="MJX-Warp is not installed.")
def test_offline_warp_renderer_honors_goal_visibility():
    """Exercise both production renderer variants on macOS Warp-CPU."""

    cpu = jax.local_devices(backend="cpu")[0]
    kwargs = {
        "num_worlds": 4,
        "image_size": (64, 64),
        "observation_type": "pixels",
        "physics_backend": "warp",
        "render": False,
        "devices": [cpu],
    }
    hidden = create_reacher(**kwargs)
    visible = create_reacher(**kwargs, visualize_goal=True)
    keys = jax.random.split(jax.random.key(31), hidden.num_worlds)
    hidden_state = hidden.reset(keys)
    visible_state = visible.reset(keys)

    def render_offline(env, data):
        camera_id = env.mj_model.camera("fixed").id
        context = mjx.create_render_context(
            mjm=env.mj_model,
            nworld=env.num_worlds,
            devices=["cpu"],
            cam_res=(64, 64),
            cam_active=[index == camera_id for index in range(env.mj_model.ncam)],
            use_textures=True,
            use_shadows=True,
            render_skybox=True,
            render_rgb=True,
            render_depth=False,
            enabled_geom_groups=[0, 1, 2],
        )
        context_pytree = context.pytree()

        @jax.jit
        def render(batch):
            return _render_batch(env.model, batch, context_pytree, camera_id)

        _, pixels = render(data)
        jax.block_until_ready(pixels)
        return pixels

    hidden_pixels = render_offline(hidden, hidden_state.data)
    visible_pixels = render_offline(visible, visible_state.data)
    assert hidden_pixels.shape == (4, 64, 64, 3)
    assert hidden_pixels.dtype == jnp.uint8
    pixels = np.asarray(hidden_pixels)
    assert np.all(pixels.reshape(4, -1).max(axis=1) > 0)
    assert np.all(pixels.reshape(4, -1).std(axis=1) > 5)
    changed = np.any(np.asarray(hidden_pixels) != np.asarray(visible_pixels), axis=-1)
    assert np.all(changed.reshape(4, -1).sum(axis=1) > 5)


def test_batched_scan_matches_repeated_steps_and_dmc_horizon():
    env = _physics_env(5, episode_length=8)
    state = env.reset(jax.random.split(jax.random.key(2), env.num_worlds))
    actions = jax.random.uniform(jax.random.key(3), (8, env.num_worlds, 2), minval=-1, maxval=1)
    final_scan, trajectory = env.rollout(state, actions)

    host_state = state
    host_rewards = []
    for action in actions:
        host_state = env.step(host_state, action)
        host_rewards.append(host_state.reward)
    host_rewards = jnp.stack(host_rewards)

    np.testing.assert_allclose(final_scan.data.qpos, host_state.data.qpos)
    np.testing.assert_allclose(final_scan.data.qvel, host_state.data.qvel)
    np.testing.assert_allclose(trajectory.reward, host_rewards)
    np.testing.assert_array_equal(trajectory.truncated[:-1], False)
    np.testing.assert_array_equal(trajectory.truncated[-1], True)
    np.testing.assert_array_equal(final_scan.step_count, 8)
    np.testing.assert_array_equal(final_scan.terminated, False)
    np.testing.assert_array_equal(trajectory.discount, 1.0)


def test_gymnax_adapters_are_batch_first():
    env = _physics_env(4)
    keys = jax.random.split(jax.random.key(0), env.num_worlds)
    obs, state = env.reset_gymnax(keys)
    output = env.step_gymnax(keys, state, jnp.zeros((env.num_worlds, 2), dtype=jnp.float32))
    next_obs, next_state, reward, done, info = output

    assert obs.shape == (4, 6)
    assert next_obs.shape == (4, 6)
    assert reward.shape == (4,)
    assert done.shape == (4,)
    assert info["distance"].shape == (4,)
    assert info["success"].shape == (4,)
    assert next_state.step_count.shape == (4,)


def test_two_device_physics_matches_single_device_in_subprocess():
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np
from envx.reacher import create_reacher

assert len(jax.local_devices()) == 2
keys = jax.random.split(jax.random.key(123), 4)
actions = jax.random.uniform(jax.random.key(456), (3, 4, 2), minval=-1, maxval=1)
single = create_reacher(4, (8, 8), observation_type='states',
                        physics_backend='jax', render=False,
                        devices=[jax.local_devices()[0]])
multi = create_reacher(4, (8, 8), observation_type='states',
                       physics_backend='jax', render=False,
                       devices=jax.local_devices())
single_final, single_traj = single.rollout(single.reset(keys), actions)
multi_final, multi_traj = multi.rollout(multi.reset(keys), actions)
jax.block_until_ready((single_final, single_traj, multi_final, multi_traj))
np.testing.assert_allclose(single_final.data.qpos,
                           multi_final.data.qpos.reshape(4, 2), atol=1e-6)
np.testing.assert_allclose(single_traj.obs, multi_traj.obs, atol=1e-6)
np.testing.assert_allclose(single_traj.reward, multi_traj.reward, atol=1e-6)
np.testing.assert_array_equal(single_traj.truncated, multi_traj.truncated)
"""
    process_env = os.environ.copy()
    process_env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    process_env["JAX_PLATFORMS"] = "cpu"
    source_dir = str(Path(__file__).parents[2] / "src")
    process_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_dir, process_env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=process_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not _warp_cuda_available(), reason="MJX-Warp rendering requires CUDA.")
def test_visual_observations_are_deterministic_and_cover_scene():
    env = create_reacher(16, (64, 64), observation_type="pixels")
    keys = jax.random.split(jax.random.key(8), env.num_worlds)
    state_a = env.reset(keys)
    state_b = env.reset(keys)
    jax.block_until_ready((state_a, state_b))

    assert state_a.obs.shape == (16, 64, 64, 3)
    assert state_a.obs.dtype == jnp.uint8
    np.testing.assert_array_equal(state_a.obs, state_b.obs)
    pixels = np.asarray(state_a.obs)
    assert np.all(pixels.reshape(16, -1).std(axis=1) > 5)


def test_visual_constructor_never_falls_back_to_cpu_renderer():
    if _warp_cuda_available():
        pytest.skip("This assertion targets non-CUDA hosts.")
    with pytest.raises(RuntimeError, match="NVIDIA CUDA GPU"):
        create_reacher(2, (64, 64), observation_type="pixels")
