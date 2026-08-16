"""Correctness checks for the optional MJX-Warp Cube environment."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from envx.cube import OGBenchCubeJaxEnv
from envx.cube._vendor.envs.cube_env import CubeEnv


@pytest.fixture(scope="module")
def env():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return OGBenchCubeJaxEnv(num_envs=2, env_type="single")


@pytest.fixture(scope="module")
def reset_state(env):
    return env.reset(jax.random.key(7), task_ids=2)


def _native_env():
    native = CubeEnv("single", ob_type="states", mode="task", visualize_info=False)
    native._mjcf_model = native.build_mjcf_model()
    native.compile_model_and_data()
    return native


def _copy_to_native(native, data):
    native.data.qpos[:] = np.asarray(data.qpos)
    native.data.qvel[:] = np.asarray(data.qvel)
    native.data.ctrl[:] = np.asarray(data.ctrl)
    native.data.mocap_pos[:] = np.asarray(data.mocap_pos)
    native.data.mocap_quat[:] = np.asarray(data.mocap_quat)
    mujoco.mj_forward(native.model, native.data)
    mujoco.mj_rnePostConstraint(native.model, native.data)


def test_reset_is_batched_deterministic_and_task_conditioned(env):
    task_ids = jnp.asarray([2, 3])
    observation_a, state_a = env.reset(jax.random.key(11), task_ids=task_ids)
    observation_b, state_b = env.reset(jax.random.key(11), task_ids=task_ids)

    assert observation_a.shape == (2, 28)
    assert state_a.goal.shape == (2, 28)
    assert state_a.data.qpos.shape == (2, env._mj_model.nq)
    np.testing.assert_array_equal(np.asarray(state_a.task_id), [2, 3])
    np.testing.assert_array_equal(np.asarray(observation_a), np.asarray(observation_b))
    np.testing.assert_array_equal(np.asarray(state_a.data.qpos), np.asarray(state_b.data.qpos))

    cube_position = np.asarray(
        state_a.data.qpos[0, env._cube_qpos_ids[0] : env._cube_qpos_ids[0] + 3]
    )
    np.testing.assert_allclose(cube_position[2], 0.02, atol=1e-7)
    np.testing.assert_allclose(cube_position[:2], [0.35, 0.0], atol=0.0101)
    np.testing.assert_allclose(
        np.asarray(state_a.data.mocap_pos[0, 0]), [0.50, 0.0, 0.02], atol=1e-7
    )


def test_renderer_visibility_matches_native_ogbench(env):
    assert env.render_geom_groups == (0, 1, 2)
    for cube_index in range(env.num_cubes):
        target_geom_id = env._mj_model.geom(f"target_object_{cube_index}").id
        assert env._mj_model.geom_group[target_geom_id] == env.hidden_render_group


def test_state_observation_matches_native_from_identical_state(env, reset_state):
    observation, state = reset_state
    native = _native_env()
    _copy_to_native(native, state.data[0])
    native.pre_step()

    np.testing.assert_allclose(
        np.asarray(observation[0]),
        native.compute_observation(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_differential_ik_control_matches_native(env, reset_state):
    _, state = reset_state
    native = _native_env()
    _copy_to_native(native, state.data[0])
    action = np.asarray([0.2, -0.3, 0.1, 0.4, -0.5], dtype=np.float32)

    native.set_control(action)
    jax_data = jax.jit(env._set_control)(state.data[0], jnp.asarray(action))
    np.testing.assert_allclose(np.asarray(jax_data.ctrl), native.data.ctrl, rtol=2e-5, atol=2e-5)


def test_mjx_dynamics_tracks_native_for_one_control_step(env, reset_state):
    _, state = reset_state
    native = _native_env()
    _copy_to_native(native, state.data[0])
    action = np.zeros(5, dtype=np.float32)
    native.set_control(action)

    jax_data = state.data[0].replace(ctrl=jnp.asarray(native.data.ctrl, dtype=jnp.float32))
    mujoco.mj_step(native.model, native.data, nstep=25)
    jax_data = jax.jit(
        lambda data: jax.lax.fori_loop(
            0, 25, lambda _, carry: mujoco.mjx.step(env._mjx_model, carry), data
        )
    )(jax_data)

    # MJX-Warp is float32 and uses a parallel solver, so compare physical
    # agreement rather than requiring bit identity with native float64 MuJoCo.
    np.testing.assert_allclose(np.asarray(jax_data.qpos), native.data.qpos, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(jax_data.qvel), native.data.qvel, rtol=2e-2, atol=2e-2)


def test_compiled_scan_matches_repeated_compiled_steps(env, reset_state):
    _, initial_state = reset_state
    actions = jax.random.uniform(jax.random.key(21), (2, 2, 5), minval=-1.0, maxval=1.0)

    state = initial_state
    observations = []
    rewards = []
    dones = []
    key = jax.random.key(22)
    for action in actions:
        key, step_key = jax.random.split(key)
        observation, state, reward, done, _ = env.step(step_key, state, action)
        observations.append(observation)
        rewards.append(reward)
        dones.append(done)

    scan_state, transitions = env.rollout(jax.random.key(23), initial_state, actions)
    np.testing.assert_allclose(
        np.asarray(transitions.observation), np.asarray(jnp.stack(observations)), atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(transitions.reward), np.asarray(jnp.stack(rewards)))
    np.testing.assert_array_equal(np.asarray(transitions.done), np.asarray(jnp.stack(dones)))
    np.testing.assert_allclose(
        np.asarray(scan_state.data.qpos), np.asarray(state.data.qpos), atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(scan_state.step_count), [2, 2])


def test_pixel_reset_and_step_use_batched_warp_renderer():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pixel_env = OGBenchCubeJaxEnv(
            num_envs=1,
            env_type="single",
            observation_type="pixels",
            image_size=(16, 16),
        )

    observation, state = pixel_env.reset(jax.random.key(31), task_ids=2)
    assert observation.shape == (1, 16, 16, 3)
    assert observation.dtype == jnp.uint8
    assert state.goal.shape == observation.shape
    assert int(observation.max()) > int(observation.min())

    next_observation, _, _, _, _ = pixel_env.step(
        jax.random.key(32), state, jnp.zeros((1, 5), dtype=jnp.float32)
    )
    jax.block_until_ready(next_observation)
    assert next_observation.shape == observation.shape
    assert next_observation.dtype == jnp.uint8
