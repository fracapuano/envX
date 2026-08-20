"""Batched DeepMind Control Suite Reacher implemented with JAX and MJX.

The model, task constants, and visual assets are loaded directly from
``dm_control.suite.reacher``.  The only structural model transformation wraps
the non-colliding target geom in a mocap body so every batched world can place
its target independently through ``mjx.Data``.  This does not change the arm's
dynamics.

The production path uses MJX-Warp physics and its batch ray renderer.  Every
world on a device is advanced and rendered by one compiled program; there is
no Python world loop or host transfer in reset, step, or rollout.
"""

from __future__ import annotations

import xml.etree.ElementTree as element_tree
from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from dm_control.suite import reacher as dmc_reacher
from mujoco import mjx

Array = jax.Array

_TARGET_SIZES = {
    "easy": float(dmc_reacher._BIG_TARGET),
    "hard": float(dmc_reacher._SMALL_TARGET),
}
_TARGET_RADIUS_RANGE = (0.05, 0.20)
_DMC_TIME_LIMIT_SECONDS = float(dmc_reacher._DEFAULT_TIME_LIMIT)
_HIDDEN_GEOM_GROUP = 5


class ReacherState(NamedTuple):
    """Complete batched environment state.

    ``data`` is simulator-private.  With multiple devices it retains its
    ``(device, worlds_per_device, ...)`` layout, while every public field is
    flattened to one leading ``num_worlds`` axis.
    """

    data: Any
    obs: Array
    reward: Array
    discount: Array
    terminated: Array
    truncated: Array
    step_count: Array
    distance: Array
    success: Array

    @property
    def done(self) -> Array:
        return jnp.logical_or(self.terminated, self.truncated)


class ReacherTransition(NamedTuple):
    """Public time-major output produced by :meth:`ReacherEnv.rollout`."""

    obs: Array
    action: Array
    reward: Array
    discount: Array
    terminated: Array
    truncated: Array
    distance: Array
    success: Array


def _dmc_model_xml(task: str) -> tuple[str, dict[str, bytes]]:
    """Load DMC Reacher and make its static target batch-addressable.

    DMC randomizes ``model.geom_pos['target']``.  A shared MJX model cannot
    hold a different value for every world, so the target is moved into a
    mocap body and randomized through ``data.mocap_pos`` instead.  The target
    has contacts disabled in the source model; this transformation is visual
    and task-state only and leaves the two-link arm dynamics unchanged.
    """

    xml, assets = dmc_reacher.get_model_and_assets()
    root = element_tree.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("DMC Reacher MJCF has no worldbody.")
    target = worldbody.find("geom[@name='target']")
    if target is None:
        raise ValueError("DMC Reacher MJCF has no target geom.")

    target_size = _TARGET_SIZES[task]
    target_position = target.attrib.pop("pos", "0 0 .01")
    target.set("pos", "0 0 0")
    target.set("size", str(target_size))
    worldbody.remove(target)
    target_body = element_tree.SubElement(
        worldbody,
        "body",
        {"name": "target_mocap", "mocap": "true", "pos": target_position},
    )
    target_body.append(target)
    return element_tree.tostring(root, encoding="unicode"), dict(assets)


def _load_dmc_model(task: str) -> mujoco.MjModel:
    xml, assets = _dmc_model_xml(task)
    return mujoco.MjModel.from_xml_string(xml, assets)


def _cuda_devices() -> list[jax.Device]:
    try:
        return list(jax.local_devices(backend="gpu"))
    except RuntimeError:
        return []


def _has_warp_cuda() -> bool:
    try:
        from mujoco.mjx.warp import io as warp_io

        return bool(warp_io.has_cuda_gpu_device())
    except (ImportError, RuntimeError):
        return False


def _render_batch(model, data, render_context, camera_id: int):
    """Render one device-local batch through the official MJX-Warp pipeline."""

    def render_one(world_data):
        world_data = mjx.refit_bvh(model, world_data, render_context)
        packed_rgb, _ = mjx.render(model, world_data, render_context)
        rgb = mjx.get_rgb(render_context, camera_id, packed_rgb)
        image = jnp.clip(jnp.rint(rgb * 255.0), 0.0, 255.0).astype(jnp.uint8)
        return world_data, image

    return jax.vmap(render_one)(data)


class ReacherEnv:
    """Fixed-size batch of DeepMind Control Suite Reacher worlds.

    Args:
        num_worlds: Total number of independent worlds.  It must be divisible
            by the selected device count.
        image_size: RGB observation size as ``(height, width)``.  DMC's native
            renderer defaults to ``(240, 320)``.
        task: DMC's ``easy`` or ``hard`` task.  These differ only in target
            radius: 0.05 m and 0.015 m respectively.
        observation_type: ``pixels`` for camera observations or ``state`` for
            DMC's flattened ``position, to_target, velocity`` observation.
        episode_length: Truncation horizon.  DMC's 20 second default at a
            0.02 second control timestep is 1000 transitions.
        physics_backend: ``warp`` for the production GPU path or ``jax`` for
            physics-only validation on any JAX device.
        render: Enable the MJX-Warp batch renderer for pixel observations.
            Rendering requires ``physics_backend='warp'`` and CUDA.  Passing
            false with pixel observations returns empty diagnostic images.
        use_shadows: Enable MJX-Warp shadow rays.  True most closely resembles
            DMC's native renderer; false trades fidelity for throughput.
        visualize_goal: Draw DMC's target ball in pixel observations.  False
            by default so pixels match datasets where the goal is task state
            but is not part of the image.  This never changes reward, success,
            state observations, or physics.
        devices: JAX devices to use.  By default visual environments use all
            local CUDA devices and non-visual environments use the default
            JAX device.

    The batch size and render resolution are static because MJX-Warp allocates
    its ray buffers when the render context is created.
    """

    action_shape = (2,)
    action_min = -1.0
    action_max = 1.0
    frame_skip = 1

    def __init__(
        self,
        num_worlds: int,
        image_size: tuple[int, int] = (240, 320),
        *,
        task: str = "easy",
        observation_type: str = "state",
        episode_length: int | None = None,
        physics_backend: str = "warp",
        render: bool | None = None,
        use_shadows: bool = True,
        visualize_goal: bool = False,
        devices: Sequence[jax.Device] | None = None,
    ) -> None:
        if num_worlds < 1:
            raise ValueError("num_worlds must be positive.")
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain two positive integers.")
        if task not in _TARGET_SIZES:
            raise ValueError("task must be either 'easy' or 'hard'.")
        if observation_type not in ("pixels", "state"):
            raise ValueError("observation_type must be 'pixels' or 'state'.")
        if render is None:
            render = observation_type == "pixels"
        if episode_length is not None and episode_length < 1:
            raise ValueError("episode_length must be positive.")
        if physics_backend not in ("jax", "warp"):
            raise ValueError("physics_backend must be either 'jax' or 'warp'.")
        if render and observation_type != "pixels":
            raise ValueError("render=True requires observation_type='pixels'.")
        if render and physics_backend != "warp":
            raise ValueError(
                "On-device rendering requires physics_backend='warp'. "
                "There is deliberately no CPU renderer fallback."
            )
        if render and (not _cuda_devices() or not _has_warp_cuda()):
            raise RuntimeError(
                "The visual Reacher environment requires MJX-Warp on an "
                "NVIDIA CUDA GPU. This host can run state-observation parity "
                "tests with observation_type='state' and "
                "physics_backend='jax'."
            )

        selected_devices = list(devices or ())
        if not selected_devices:
            selected_devices = _cuda_devices() if render else [jax.devices()[0]]
        if not selected_devices:
            raise RuntimeError("No compatible JAX devices are available.")
        if render and any(device.platform != "gpu" for device in selected_devices):
            raise ValueError("Visual environments can only use CUDA devices.")
        if num_worlds % len(selected_devices):
            raise ValueError(
                f"num_worlds ({num_worlds}) must be divisible by the device "
                f"count ({len(selected_devices)})."
            )

        self.num_worlds = int(num_worlds)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.task = task
        self.observation_type = observation_type
        self.physics_backend = physics_backend
        self.render_enabled = bool(render)
        self.use_shadows = bool(use_shadows)
        self.visualize_goal = bool(visualize_goal)
        self.devices = tuple(selected_devices)
        self.num_devices = len(self.devices)
        self.worlds_per_device = self.num_worlds // self.num_devices

        self.mj_model = _load_dmc_model(task)
        self._target_geom_id = int(self.mj_model.geom("target").id)
        if not self.visualize_goal:
            self.mj_model.geom_group[self._target_geom_id] = _HIDDEN_GEOM_GROUP
        if episode_length is None:
            episode_length = round(_DMC_TIME_LIMIT_SECONDS / float(self.mj_model.opt.timestep))
        self.episode_length = int(episode_length)
        model_device = self.devices[0] if self.num_devices == 1 else None
        self.model = mjx.put_model(self.mj_model, impl=physics_backend, device=model_device)
        self._qpos0 = jnp.asarray(self.mj_model.qpos0)
        self._qvel0 = jnp.zeros((self.mj_model.nv,), dtype=self._qpos0.dtype)
        self._wrist_range = jnp.asarray(self.mj_model.jnt_range[self.mj_model.joint("wrist").id])
        self._finger_geom_id = int(self.mj_model.geom("finger").id)
        target_body_id = int(self.mj_model.geom_bodyid[self._target_geom_id])
        self._target_mocap_id = int(self.mj_model.body_mocapid[target_body_id])
        self._target_z = float(self.mj_model.body_pos[target_body_id, 2])
        self._success_radius = float(
            self.mj_model.geom_size[self._target_geom_id, 0]
            + self.mj_model.geom_size[self._finger_geom_id, 0]
        )
        self._camera_id = int(self.mj_model.camera("fixed").id)

        self._render_context = None
        self._render_context_pytree = None
        if self.render_enabled:
            height, width = self.image_size
            context_devices = [f"cuda:{device.local_hardware_id}" for device in self.devices]
            active_cameras = [
                camera_id == self._camera_id for camera_id in range(self.mj_model.ncam)
            ]
            self._render_context = mjx.create_render_context(
                mjm=self.mj_model,
                nworld=self.worlds_per_device,
                devices=context_devices,
                cam_res=(width, height),
                cam_active=active_cameras,
                use_textures=True,
                use_shadows=self.use_shadows,
                render_skybox=True,
                render_rgb=True,
                render_depth=False,
                enabled_geom_groups=[0, 1, 2],
            )
            self._render_context_pytree = self._render_context.pytree()

        if self.num_devices == 1:
            self._compiled_reset = jax.jit(self._reset_local)
            self._compiled_step = jax.jit(self._step_local)
            self._compiled_rollout = jax.jit(self._rollout_local)
        else:
            self._compiled_reset = jax.pmap(
                self._reset_local, devices=self.devices, axis_name="device"
            )
            self._compiled_step = jax.pmap(
                self._step_local, devices=self.devices, axis_name="device"
            )
            self._compiled_rollout = jax.pmap(
                self._rollout_local, devices=self.devices, axis_name="device"
            )

    @property
    def observation_shape(self) -> tuple[int, ...]:
        if self.observation_type == "state":
            return (6,)
        return (*self.image_size, 3)

    @property
    def dt(self) -> float:
        return float(self.mj_model.opt.timestep * self.frame_skip)

    @property
    def time_limit(self) -> float:
        return self.episode_length * self.dt

    def _make_data(self):
        kwargs: dict[str, int] = {}
        if self.physics_backend == "warp":
            kwargs = {"naconmax": 1, "njmax": 8}
        return mjx.make_data(self.mj_model, impl=self.physics_backend, **kwargs)

    def _reset_one(self, key: Array):
        shoulder_key, wrist_key, angle_key, radius_key = jax.random.split(key, 4)
        shoulder = jax.random.uniform(shoulder_key, (), minval=-jnp.pi, maxval=jnp.pi)
        wrist = jax.random.uniform(
            wrist_key,
            (),
            minval=self._wrist_range[0],
            maxval=self._wrist_range[1],
        )
        qpos = self._qpos0.at[:].set(jnp.stack((shoulder, wrist)))

        angle = jax.random.uniform(angle_key, (), minval=0.0, maxval=2.0 * jnp.pi)
        radius = jax.random.uniform(
            radius_key,
            (),
            minval=_TARGET_RADIUS_RANGE[0],
            maxval=_TARGET_RADIUS_RANGE[1],
        )
        target_position = jnp.asarray(
            [radius * jnp.sin(angle), radius * jnp.cos(angle), self._target_z]
        )

        data = self._make_data().replace(qpos=qpos, qvel=self._qvel0)
        data = data.replace(mocap_pos=data.mocap_pos.at[self._target_mocap_id].set(target_position))
        return mjx.forward(self.model, data)

    def _state_observation_one(self, data) -> Array:
        to_target = (
            data.geom_xpos[self._target_geom_id, :2] - data.geom_xpos[self._finger_geom_id, :2]
        )
        return jnp.concatenate((data.qpos, to_target, data.qvel))

    def _observe(self, data):
        if self.observation_type == "state":
            return data, jax.vmap(self._state_observation_one)(data)
        if not self.render_enabled:
            empty = jnp.zeros((self.worlds_per_device, 0, 0, 3), dtype=jnp.uint8)
            return data, empty
        return _render_batch(
            self.model,
            data,
            self._render_context_pytree,
            self._camera_id,
        )

    def _metrics(self, data) -> tuple[Array, Array, Array]:
        displacement = (
            data.geom_xpos[:, self._target_geom_id, :2]
            - data.geom_xpos[:, self._finger_geom_id, :2]
        )
        distance = jnp.linalg.norm(displacement, axis=-1)
        success = distance <= self._success_radius
        return success.astype(jnp.float32), distance, success

    def _reset_local(self, keys: Array) -> ReacherState:
        data = jax.vmap(self._reset_one)(keys)
        _, distance, success = self._metrics(data)
        data, obs = self._observe(data)
        zeros = jnp.zeros((self.worlds_per_device,), dtype=jnp.float32)
        ones = jnp.ones((self.worlds_per_device,), dtype=jnp.float32)
        false = jnp.zeros((self.worlds_per_device,), dtype=jnp.bool_)
        return ReacherState(
            data=data,
            obs=obs,
            reward=zeros,
            discount=ones,
            terminated=false,
            truncated=false,
            step_count=jnp.zeros((self.worlds_per_device,), dtype=jnp.int32),
            distance=distance,
            success=success,
        )

    def _step_one(self, data, action: Array):
        data = mjx.step(self.model, data.replace(ctrl=action))
        # MJX leaves position-dependent fields at the pre-integration state.
        # DMC observes and rewards the post-transition state, so refresh the
        # kinematics after integration just as native MuJoCo exposes them.
        return mjx.kinematics(self.model, data)

    def _step_local(self, data, step_count: Array, action: Array) -> ReacherState:
        data = jax.vmap(self._step_one)(data, action)
        reward, distance, success = self._metrics(data)
        step_count = step_count + 1
        terminated = jnp.zeros_like(step_count, dtype=jnp.bool_)
        truncated = step_count >= self.episode_length
        discount = jnp.ones_like(reward)
        data, obs = self._observe(data)
        return ReacherState(
            data=data,
            obs=obs,
            reward=reward,
            discount=discount,
            terminated=terminated,
            truncated=truncated,
            step_count=step_count,
            distance=distance,
            success=success,
        )

    def _rollout_local(
        self, data, step_count: Array, actions: Array
    ) -> tuple[ReacherState, ReacherTransition]:
        def scan_step(carry, action):
            next_state = self._step_local(carry[0], carry[1], action)
            transition = ReacherTransition(
                obs=next_state.obs,
                action=action,
                reward=next_state.reward,
                discount=next_state.discount,
                terminated=next_state.terminated,
                truncated=next_state.truncated,
                distance=next_state.distance,
                success=next_state.success,
            )
            return (next_state.data, next_state.step_count), transition

        (data, step_count), trajectory = jax.lax.scan(scan_step, (data, step_count), actions)
        final_state = ReacherState(
            data=data,
            obs=trajectory.obs[-1],
            reward=trajectory.reward[-1],
            discount=trajectory.discount[-1],
            terminated=trajectory.terminated[-1],
            truncated=trajectory.truncated[-1],
            step_count=step_count,
            distance=trajectory.distance[-1],
            success=trajectory.success[-1],
        )
        return final_state, trajectory

    def _reshape_keys_for_devices(self, keys: Array) -> Array:
        return keys.reshape(self.num_devices, self.worlds_per_device, *keys.shape[1:])

    def _flatten_public(self, value: Array) -> Array:
        return value.reshape(self.num_worlds, *value.shape[2:])

    def _public_state(self, state: ReacherState) -> ReacherState:
        if self.num_devices == 1:
            return state
        return ReacherState(
            data=state.data,
            obs=self._flatten_public(state.obs),
            reward=self._flatten_public(state.reward),
            discount=self._flatten_public(state.discount),
            terminated=self._flatten_public(state.terminated),
            truncated=self._flatten_public(state.truncated),
            step_count=self._flatten_public(state.step_count),
            distance=self._flatten_public(state.distance),
            success=self._flatten_public(state.success),
        )

    def reset(self, keys: Array) -> ReacherState:
        """Reset all worlds from one JAX PRNG key per world."""

        if keys.shape[0] != self.num_worlds:
            raise ValueError(f"Expected {self.num_worlds} keys, got shape {keys.shape}.")
        if self.num_devices == 1:
            return self._compiled_reset(keys)
        state = self._compiled_reset(self._reshape_keys_for_devices(keys))
        return self._public_state(state)

    def sample_actions(self, key: Array) -> Array:
        """Sample one DMC-range action independently for every world."""

        return jax.random.uniform(
            key,
            (self.num_worlds, *self.action_shape),
            minval=self.action_min,
            maxval=self.action_max,
        )

    def step(self, state: ReacherState, actions: Array) -> ReacherState:
        """Advance every world by one DMC control step and observe it."""

        expected = (self.num_worlds, 2)
        if actions.shape != expected:
            raise ValueError(f"Expected actions with shape {expected}.")
        if self.num_devices == 1:
            return self._compiled_step(state.data, state.step_count, actions)
        sharded_actions = actions.reshape(self.num_devices, self.worlds_per_device, 2)
        sharded_count = state.step_count.reshape(self.num_devices, self.worlds_per_device)
        next_state = self._compiled_step(state.data, sharded_count, sharded_actions)
        return self._public_state(next_state)

    def rollout(
        self, state: ReacherState, actions: Array
    ) -> tuple[ReacherState, ReacherTransition]:
        """Run a time-major action sequence with one compiled ``lax.scan``."""

        if actions.ndim != 3 or actions.shape[1:] != (self.num_worlds, 2):
            raise ValueError(
                f"Expected actions with shape (time, num_worlds, 2), got {actions.shape}."
            )
        if actions.shape[0] < 1:
            raise ValueError("A rollout must contain at least one action.")
        if self.num_devices == 1:
            return self._compiled_rollout(state.data, state.step_count, actions)

        local_actions = actions.reshape(
            actions.shape[0], self.num_devices, self.worlds_per_device, 2
        ).swapaxes(0, 1)
        local_count = state.step_count.reshape(self.num_devices, self.worlds_per_device)
        final_state, trajectory = self._compiled_rollout(state.data, local_count, local_actions)
        final_state = self._public_state(final_state)

        def flatten_trajectory(value):
            value = value.swapaxes(0, 1)
            return value.reshape(value.shape[0], self.num_worlds, *value.shape[3:])

        trajectory = jax.tree.map(flatten_trajectory, trajectory)
        return final_state, trajectory

    def reset_gymnax(self, keys: Array) -> tuple[Array, ReacherState]:
        state = self.reset(keys)
        return state.obs, state

    def step_gymnax(
        self, keys: Array, state: ReacherState, actions: Array
    ) -> tuple[Array, ReacherState, Array, Array, dict[str, Array]]:
        """Gymnax-shaped batched step; keys are unused by deterministic steps."""

        del keys
        state = self.step(state, actions)
        info = {
            "discount": state.discount,
            "distance": state.distance,
            "success": state.success,
            "terminated": state.terminated,
            "truncated": state.truncated,
        }
        return state.obs, state, state.reward, state.done, info


def create_reacher(
    num_worlds: int,
    image_size: tuple[int, int] = (240, 320),
    **kwargs: Any,
) -> ReacherEnv:
    """Create a persistent batch of DMC Reacher worlds."""

    return ReacherEnv(num_worlds=num_worlds, image_size=image_size, **kwargs)


__all__ = (
    "ReacherEnv",
    "ReacherState",
    "ReacherTransition",
    "create_reacher",
)
