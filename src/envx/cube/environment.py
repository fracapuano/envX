"""Batched, functional OGBench Cube environments backed by MJX-Warp.

The original :class:`CubeEnv` remains the reference CPU/Gymnasium
implementation.  This module builds the same MuJoCo model once on the host,
then keeps every rollout state and all transition work in JAX.  A batch is a
first-class part of the API: ``num_envs`` worlds are stepped by one compiled
call rather than by a Python vector-environment loop.

MJX-Warp is intentional here.  The Cube model contains cylinder--mesh contact
pairs that are not implemented by the MJX-JAX collision backend.
"""

from __future__ import annotations

import os
import platform
import warnings
from typing import Any, NamedTuple

# Warp probes CUDA during import.  Apple machines have no CUDA device, so make
# its CPU validation backend explicit before importing MJX.
if platform.system() == "Darwin":
    os.environ.setdefault("WARP_DISABLE_CUDA", "1")

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gymnasium.spaces import Box
    from mujoco import mjx
except ImportError as exc:  # pragma: no cover - exercised in a base-only install.
    raise ImportError(
        "envX Cube requires JAX and MJX-Warp. Install envX with its standard dependencies."
    ) from exc

from envx.cube._vendor.envs.cube_env import CubeEnv

_NUM_CUBES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "quadruple": 4,
    "octuple": 8,
}

_HORIZONS = {
    "single": 200,
    "double": 500,
    "triple": 1000,
    "quadruple": 1000,
    "octuple": 1500,
}


class CubeJaxParams(NamedTuple):
    """Dynamic environment parameters accepted by ``reset`` and ``step``."""

    max_episode_steps: jax.Array
    success_threshold: jax.Array


class CubeJaxState(NamedTuple):
    """Complete batched simulator state.

    ``data`` is an MJX pytree whose public leaves have a leading ``num_envs``
    axis.  Keeping it in the state makes the environment functional and safe
    to pass through ``jax.jit``, ``jax.lax.scan``, and RL learner state.
    """

    data: Any
    goal: jax.Array
    task_id: jax.Array
    step_count: jax.Array
    success: jax.Array


class CubeJaxTransition(NamedTuple):
    """Time-major outputs produced by :meth:`OGBenchCubeJaxEnv.rollout`."""

    observation: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict[str, jax.Array]


def _quat_mul(q0: jax.Array, q1: jax.Array) -> jax.Array:
    """Hamilton product for MuJoCo ``(w, x, y, z)`` quaternions."""

    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return jnp.asarray(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def _quat_conjugate(q: jax.Array) -> jax.Array:
    return q * jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def _quat_rotate(q: jax.Array, vector: jax.Array) -> jax.Array:
    """Rotate a vector without constructing a rotation matrix."""

    qvec = q[1:]
    uv = jnp.cross(qvec, vector)
    uuv = jnp.cross(qvec, uv)
    return vector + 2.0 * (q[0] * uv + uuv)


def _quat_to_rotvec(q: jax.Array) -> jax.Array:
    """MuJoCo-compatible shortest-axis quaternion logarithm."""

    q = q / jnp.maximum(jnp.linalg.norm(q), jnp.finfo(q.dtype).tiny)
    q = jnp.where(q[0] < 0.0, -q, q)
    vector_norm = jnp.linalg.norm(q[1:])
    angle = 2.0 * jnp.arctan2(vector_norm, jnp.maximum(q[0], 0.0))
    scale = jnp.where(vector_norm > 1e-8, angle / vector_norm, 2.0)
    return q[1:] * scale


def _yaw_quat(yaw: jax.Array) -> jax.Array:
    half_yaw = 0.5 * yaw
    return jnp.asarray([jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)])


def _quat_yaw(q: jax.Array) -> jax.Array:
    w, x, y, z = q
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class OGBenchCubeJaxEnv:
    """Massively batched OGBench Cube simulation with a Gymnax-style API.

    Args:
        num_envs: Number of worlds in the compiled batch.
        env_type: ``single``, ``double``, ``triple``, ``quadruple``, or
            ``octuple``.
        observation_type: ``states`` for dataset-compatible state vectors or
            ``pixels`` for batched MJX-Warp RGB observations.
        reward_task_id: ``None`` selects goal-conditioned sparse rewards.  An
            integer in ``1..5`` selects the corresponding single-task reward;
            ``0`` selects OGBench's default task 2.
        permute_blocks: Match OGBench's task reset permutation.  Defaults to
            true for goal-conditioned tasks and false for single-task tasks.
        success_timing: ``pre`` or ``post`` action, matching ``CubeEnv``.
            Defaults to ``post`` for goal-conditioned and ``pre`` for
            single-task rewards.
        terminate_at_goal: End a world when all cubes meet their goals.
        use_oracle_rep: Store only scaled cube positions in ``state.goal``,
            matching ``CubeEnv(use_oracle_rep=True)``.
        image_size: ``(height, width)`` for pixel observations.
        device: Optional JAX device.  A CUDA device is strongly recommended.
        naconmax: MJX-Warp contact capacity shared by the batch.
        njmax: MJX-Warp constraint capacity shared by the batch.

    The public methods follow Gymnax's functional convention::

        observation, state = env.reset(key, params)
        observation, state, reward, done, info = env.step(
            key, state, action, params
        )

    Unlike a scalar Gymnax environment, observations, actions, rewards, and
    done flags always carry a leading ``num_envs`` axis.
    """

    name = "ogbench-cube-jax"
    render_geom_groups = (0, 1, 2)
    hidden_render_group = 5

    def __init__(
        self,
        num_envs: int,
        env_type: str = "single",
        observation_type: str = "states",
        reward_task_id: int | None = None,
        permute_blocks: bool | None = None,
        success_timing: str | None = None,
        terminate_at_goal: bool = True,
        use_oracle_rep: bool = False,
        image_size: tuple[int, int] = (64, 64),
        device: jax.Device | None = None,
        naconmax: int | None = None,
        njmax: int | None = None,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if env_type not in _NUM_CUBES:
            raise ValueError(f"Unknown env_type {env_type!r}; expected one of {tuple(_NUM_CUBES)}.")
        if observation_type not in ("states", "pixels"):
            raise ValueError("observation_type must be either 'states' or 'pixels'.")
        if reward_task_id is not None and not 0 <= reward_task_id <= 5:
            raise ValueError("reward_task_id must be None or an integer in [0, 5].")

        is_single_task = reward_task_id is not None
        if reward_task_id == 0:
            reward_task_id = 2
        if permute_blocks is None:
            permute_blocks = not is_single_task
        if success_timing is None:
            success_timing = "pre" if is_single_task else "post"
        if success_timing not in ("pre", "post"):
            raise ValueError("success_timing must be either 'pre' or 'post'.")

        self.num_envs = int(num_envs)
        self.env_type = env_type
        self.num_cubes = _NUM_CUBES[env_type]
        self.observation_type = observation_type
        self.reward_task_id = reward_task_id
        self.permute_blocks = bool(permute_blocks)
        self.success_timing = success_timing
        self.terminate_at_goal = bool(terminate_at_goal)
        self.use_oracle_rep = bool(use_oracle_rep)
        self.image_size = tuple(int(size) for size in image_size)
        self.device = device or jax.devices()[0]

        # Use CubeEnv only as the canonical MJCF/model builder.  No native
        # environment reset or native physics occurs on the rollout path.
        template = CubeEnv(
            env_type=env_type,
            ob_type="pixels" if observation_type == "pixels" else "states",
            mode="task",
            visualize_info=False,
            pixel_transparent_arm=True,
            width=self.image_size[1],
            height=self.image_size[0],
        )
        template._mjcf_model = template.build_mjcf_model()
        template.compile_model_and_data()

        # Match CubeEnv's episode colors.  The reference calls this from reset,
        # after model compilation, so it must be applied before put_model.
        for cube_index in range(self.num_cubes):
            for geom_id in template._cube_geom_ids_list[cube_index]:
                template.model.geom_rgba[geom_id] = template._cube_colors[cube_index]
            for geom_id in template._cube_target_geom_ids_list[cube_index]:
                template.model.geom_rgba[geom_id, :3] = template._cube_colors[cube_index, :3]
                template.model.geom_rgba[geom_id, 3] = 0.0
                # The native renderer honors alpha=0, while MJX-Warp's batch
                # raycaster does not provide equivalent alpha compositing.
                # Exclude hidden task markers structurally from its BVH.
                template.model.geom_group[geom_id] = self.hidden_render_group

        self._mj_model = template.model
        self._mjx_model = mjx.put_model(self._mj_model, device=self.device, impl="warp")

        # The original controller solves IK in a separate, arm-only model.
        # Converting that model to MJX-JAX gives the same kinematic structure
        # without running collision dynamics inside each of the 20 iterations.
        self._ik_mj_model = template._ik._model
        self._ik_model = mjx.put_model(self._ik_mj_model, device=self.device, impl="jax")
        self._ik_data = mjx.make_data(self._ik_mj_model, device=self.device, impl="jax")

        if naconmax is None:
            naconmax = max(512, 32 * self.num_envs)
        if njmax is None:
            njmax = max(1024, 128 * self.num_envs)
        self.naconmax = int(naconmax)
        self.njmax = int(njmax)
        self._data_template = mjx.make_data(
            self._mj_model,
            device=self.device,
            impl="warp",
            naconmax=self.naconmax,
            njmax=self.njmax,
        )

        self._arm_qpos_ids = jnp.asarray(self._mj_model.jnt_qposadr[template._arm_joint_ids])
        self._arm_dof_ids = jnp.asarray(self._mj_model.jnt_dofadr[template._arm_joint_ids])
        self._arm_actuator_ids = jnp.asarray(template._arm_actuator_ids)
        self._gripper_actuator_ids = jnp.asarray(template._gripper_actuator_ids)
        self._gripper_qpos_id = int(self._mj_model.jnt_qposadr[template._gripper_opening_joint_id])
        self._pinch_site_id = int(template._pinch_site_id)
        self._right_pad_body_id = int(self._mj_model.body("ur5e/robotiq/right_pad").id)

        self._cube_qpos_ids = jnp.asarray(
            [
                self._mj_model.joint(f"object_joint_{index}").qposadr[0]
                for index in range(self.num_cubes)
            ]
        )
        self._cube_target_mocap_ids = jnp.asarray(template._cube_target_mocap_ids)
        self._cube_qpos_offsets = jnp.arange(7)

        self._home_qpos = jnp.asarray(template._home_qpos)
        self._workspace_low = jnp.asarray(template._workspace_bounds[0])
        self._workspace_high = jnp.asarray(template._workspace_bounds[1])
        self._arm_sampling_low = jnp.asarray(template._arm_sampling_bounds[0])
        self._arm_sampling_high = jnp.asarray(template._arm_sampling_bounds[1])
        self._down_quat = jnp.asarray(template._effector_down_rotation.wxyz)
        self._pinch_to_attach_quat = jnp.asarray(template._T_pa.rotation().wxyz)
        self._pinch_to_attach_pos = jnp.asarray(template._T_pa.translation())

        self._ik_site_id = int(self._ik_mj_model.site("attachment_site").id)
        self._ik_site_body_id = int(self._ik_mj_model.site_bodyid[self._ik_site_id])
        self._ik_site_quat = jnp.asarray(self._ik_mj_model.site_quat[self._ik_site_id])

        self._task_init_xyzs = jnp.asarray(
            np.stack([task["init_xyzs"] for task in template.task_infos])
        )
        self._task_goal_xyzs = jnp.asarray(
            np.stack([task["goal_xyzs"] for task in template.task_infos])
        )
        self.task_names = tuple(task["task_name"] for task in template.task_infos)

        self.default_params = CubeJaxParams(
            max_episode_steps=jnp.asarray(_HORIZONS[env_type], dtype=jnp.int32),
            success_threshold=jnp.asarray(0.04, dtype=jnp.float32),
        )

        self._render_context = None
        if observation_type == "pixels":
            camera_id = int(self._mj_model.camera("front_pixels").id)
            active_cameras = [index == camera_id for index in range(self._mj_model.ncam)]
            render_device = None
            if self.device.platform == "gpu":
                render_device = [f"cuda:{self.device.id}"]
            self._render_context = mjx.create_render_context(
                self._mj_model,
                nworld=self.num_envs,
                devices=render_device,
                cam_res=(self.image_size[1], self.image_size[0]),
                cam_active=active_cameras,
                render_rgb=True,
                render_depth=False,
                render_seg=False,
                # Match MuJoCo's default MjvOption: groups 0--2 are visible;
                # group 3 contains translucent walls/collision geometry that
                # otherwise becomes an opaque foreground obstruction in Warp.
                enabled_geom_groups=list(self.render_geom_groups),
                use_textures=True,
                use_shadows=False,
            )

        if self.device.platform != "gpu":
            warnings.warn(
                "ogbench-cube-jax is running on the MJX-Warp CPU validation backend. "
                "This is useful for correctness checks on macOS, but fast batched "
                "simulation requires a CUDA GPU.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Compile one whole batch.  The scan version below calls the unwrapped
        # implementations so XLA/Warp can fuse and capture the complete rollout.
        self._compiled_reset = jax.jit(self._reset_impl)
        self._compiled_step = jax.jit(self._step_impl)
        self._compiled_rollout = jax.jit(self._rollout_impl)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        if self.observation_type == "pixels":
            return (*self.image_size, 3)
        return (19 + 9 * self.num_cubes,)

    def action_space(self, params: CubeJaxParams | None = None) -> Box:
        del params
        return Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def observation_space(self, params: CubeJaxParams | None = None) -> Box:
        del params
        if self.observation_type == "pixels":
            return Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        return Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

    def sample_actions(self, key: jax.Array) -> jax.Array:
        """Sample one normalized action per world."""

        return jax.random.uniform(key, (self.num_envs, 5), minval=-1.0, maxval=1.0)

    def reset(
        self,
        key: jax.Array,
        params: CubeJaxParams | None = None,
        task_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, CubeJaxState]:
        """Reset all worlds.

        ``task_ids`` may be a scalar or an array of shape ``(num_envs,)``.
        IDs are one-based, as in OGBench.  Zero asks each world to sample one
        of the five tasks independently.
        """

        del params  # Reset distributions do not depend on dynamic parameters.
        if task_ids is None:
            task_ids = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        else:
            task_ids = jnp.asarray(task_ids, dtype=jnp.int32)
            if task_ids.ndim == 0:
                task_ids = jnp.broadcast_to(task_ids, (self.num_envs,))
            if task_ids.shape != (self.num_envs,):
                raise ValueError(
                    f"task_ids must have shape ({self.num_envs},), got {task_ids.shape}."
                )
        if self.reward_task_id is not None:
            task_ids = jnp.full((self.num_envs,), self.reward_task_id, dtype=jnp.int32)
        return self._compiled_reset(key, task_ids)

    def step(
        self,
        key: jax.Array,
        state: CubeJaxState,
        action: jax.Array,
        params: CubeJaxParams | None = None,
    ) -> tuple[jax.Array, CubeJaxState, jax.Array, jax.Array, dict[str, jax.Array]]:
        """Advance every world by one OGBench control step (25 physics steps)."""

        action = jnp.asarray(action)
        if action.shape != (self.num_envs, 5):
            raise ValueError(f"action must have shape ({self.num_envs}, 5), got {action.shape}.")
        return self._compiled_step(key, state, action, params or self.default_params)

    def rollout(
        self,
        key: jax.Array,
        state: CubeJaxState,
        actions: jax.Array,
        params: CubeJaxParams | None = None,
    ) -> tuple[CubeJaxState, CubeJaxTransition]:
        """Run a time-major ``(T, num_envs, 5)`` action tensor in one scan."""

        actions = jnp.asarray(actions)
        if actions.ndim != 3 or actions.shape[1:] != (self.num_envs, 5):
            raise ValueError(
                f"actions must have shape (T, {self.num_envs}, 5), got {actions.shape}."
            )
        return self._compiled_rollout(key, state, actions, params or self.default_params)

    def _solve_ik(
        self, target_pos: jax.Array, target_quat: jax.Array, current_qpos: jax.Array
    ) -> jax.Array:
        """Twenty fixed-shape differential-IK iterations, matching CubeEnv."""

        def iteration(_, qpos):
            data = self._ik_data.replace(qpos=qpos, qvel=jnp.zeros_like(self._ik_data.qvel))
            data = mjx.kinematics(self._ik_model, data)
            data = mjx.com_pos(self._ik_model, data)

            current_pos = data.site_xpos[self._ik_site_id]
            current_quat = _quat_mul(data.xquat[self._ik_site_body_id], self._ik_site_quat)
            pos_error = target_pos - current_pos
            rot_error = _quat_to_rotvec(_quat_mul(target_quat, _quat_conjugate(current_quat)))
            error = jnp.concatenate([pos_error, rot_error])

            jac_pos, jac_rot = mjx.jac(
                self._ik_model,
                data,
                current_pos,
                jnp.asarray(self._ik_site_body_id),
            )
            jacobian = jnp.concatenate([jac_pos.T, jac_rot.T], axis=0)
            damping = 1e-6 if qpos.dtype == jnp.float32 else 1e-12
            update = jacobian.T @ jnp.linalg.solve(
                jacobian @ jacobian.T + damping * jnp.eye(6, dtype=qpos.dtype),
                error,
            )
            update_max = jnp.max(jnp.abs(update))
            update *= jnp.minimum(1.0, jnp.deg2rad(45.0) / jnp.maximum(update_max, 1e-12))
            reached = (jnp.linalg.norm(pos_error) <= 1e-4) & (jnp.linalg.norm(rot_error) <= 1e-4)
            return jnp.where(reached, qpos, qpos + update)

        return jax.lax.fori_loop(0, 20, iteration, current_qpos)

    def _pinch_to_attach(
        self, pinch_pos: jax.Array, pinch_quat: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        attach_pos = pinch_pos + _quat_rotate(pinch_quat, self._pinch_to_attach_pos)
        attach_quat = _quat_mul(pinch_quat, self._pinch_to_attach_quat)
        return attach_pos, attach_quat

    def _initialize_arm(self, data: Any, key: jax.Array) -> Any:
        pos_key, yaw_key = jax.random.split(key)
        pinch_pos = jax.random.uniform(
            pos_key, (3,), minval=self._arm_sampling_low, maxval=self._arm_sampling_high
        )
        yaw = jax.random.uniform(yaw_key, (), minval=-jnp.pi, maxval=jnp.pi)
        pinch_quat = _quat_mul(_yaw_quat(yaw), self._down_quat)
        target_pos, target_quat = self._pinch_to_attach(pinch_pos, pinch_quat)
        arm_qpos = self._solve_ik(target_pos, target_quat, self._home_qpos)
        return data.replace(qpos=data.qpos.at[self._arm_qpos_ids].set(arm_qpos))

    def _set_control(self, data: Any, action: jax.Array) -> Any:
        # The normalized OGBench action ranges are [0.05m, 0.3rad, 1.0].
        scaled_action = action * jnp.asarray([0.05, 0.05, 0.05, 0.3, 1.0])
        pinch_pos = data.site_xpos[self._pinch_site_id]
        pinch_matrix = data.site_xmat[self._pinch_site_id]
        pinch_yaw = jnp.arctan2(pinch_matrix[1, 0], pinch_matrix[0, 0])
        gripper_opening = jnp.clip(data.qpos[self._gripper_qpos_id] / 0.8, 0.0, 1.0)

        target_pinch_pos = jnp.clip(
            pinch_pos + scaled_action[:3], self._workspace_low, self._workspace_high
        )
        target_yaw = jnp.clip(pinch_yaw + scaled_action[3], -jnp.pi, jnp.pi)
        target_pinch_quat = _quat_mul(_yaw_quat(target_yaw), self._down_quat)
        target_gripper = jnp.clip(gripper_opening + scaled_action[4], 0.0, 1.0)

        target_pos, target_quat = self._pinch_to_attach(target_pinch_pos, target_pinch_quat)
        arm_qpos = self._solve_ik(target_pos, target_quat, data.qpos[self._arm_qpos_ids])
        ctrl = data.ctrl.at[self._arm_actuator_ids].set(arm_qpos)
        ctrl = ctrl.at[self._gripper_actuator_ids].set(255.0 * target_gripper)
        return data.replace(ctrl=ctrl)

    def _physics_step(self, data: Any, action: jax.Array) -> Any:
        data = self._set_control(data, action)
        return jax.lax.fori_loop(0, 25, lambda _, carry: mjx.step(self._mjx_model, carry), data)

    def _cube_success(self, data: Any, threshold: jax.Array) -> jax.Array:
        cube_pos = data.qpos[self._cube_qpos_ids[:, None] + jnp.arange(3)]
        target_pos = data.mocap_pos[self._cube_target_mocap_ids]
        return jnp.linalg.norm(cube_pos - target_pos, axis=-1) <= threshold

    def _state_observation(self, data: Any) -> jax.Array:
        xyz_center = jnp.asarray([0.425, 0.0, 0.0])
        pinch_pos = data.site_xpos[self._pinch_site_id]
        pinch_matrix = data.site_xmat[self._pinch_site_id]
        pinch_yaw = jnp.arctan2(pinch_matrix[1, 0], pinch_matrix[0, 0])
        opening = jnp.clip(data.qpos[self._gripper_qpos_id] / 0.8, 0.0, 1.0)
        contact = jnp.clip(
            jnp.linalg.norm(data._impl.cfrc_ext[self._right_pad_body_id]) / 50.0, 0.0, 1.0
        )

        parts = [
            data.qpos[self._arm_qpos_ids],
            data.qvel[self._arm_dof_ids],
            (pinch_pos - xyz_center) * 10.0,
            jnp.asarray([jnp.cos(pinch_yaw)]),
            jnp.asarray([jnp.sin(pinch_yaw)]),
            jnp.asarray([opening * 3.0]),
            jnp.asarray([contact]),
        ]
        cube_qpos = data.qpos[self._cube_qpos_ids[:, None] + self._cube_qpos_offsets]
        cube_pos = cube_qpos[:, :3]
        cube_quat = cube_qpos[:, 3:]
        cube_yaw = jax.vmap(_quat_yaw)(cube_quat)
        cube_observation = jnp.concatenate(
            [
                (cube_pos - xyz_center) * 10.0,
                cube_quat,
                jnp.cos(cube_yaw)[:, None],
                jnp.sin(cube_yaw)[:, None],
            ],
            axis=-1,
        )
        parts.append(cube_observation.reshape(-1))
        return jnp.concatenate(parts)

    def _oracle_observation(self, data: Any) -> jax.Array:
        xyz_center = jnp.asarray([0.425, 0.0, 0.0])
        cube_pos = data.qpos[self._cube_qpos_ids[:, None] + jnp.arange(3)]
        return ((cube_pos - xyz_center) * 10.0).reshape(-1)

    def _render_one(self, data: Any, render_context: Any) -> tuple[jax.Array, Any]:
        data = mjx.refit_bvh(self._mjx_model, data, render_context)
        packed_rgb, _ = mjx.render(self._mjx_model, data, render_context)
        rgb = mjx.get_rgb(render_context, 0, packed_rgb)
        image = jnp.clip(jnp.rint(rgb * 255.0), 0.0, 255.0).astype(jnp.uint8)
        return image, data

    def _observe_batch(self, data: Any) -> tuple[jax.Array, Any]:
        if self.observation_type == "states":
            return jax.vmap(self._state_observation)(data), data
        return jax.vmap(self._render_one, in_axes=(0, None))(data, self._render_context.pytree())

    def _reset_one(
        self, key: jax.Array, requested_task_id: jax.Array
    ) -> tuple[Any, Any, jax.Array]:
        task_key, permutation_key, goal_arm_key, goal_action_key, arm_key, cube_key = (
            jax.random.split(key, 6)
        )
        sampled_task_id = jax.random.randint(task_key, (), 1, 6)
        task_id = jnp.where(requested_task_id == 0, sampled_task_id, requested_task_id)

        permutation = jax.random.permutation(permutation_key, self.num_cubes)
        if not self.permute_blocks:
            permutation = jnp.arange(self.num_cubes)
        init_xyzs = self._task_init_xyzs[task_id - 1][permutation]
        goal_xyzs = self._task_goal_xyzs[task_id - 1][permutation]

        # Build the goal observation exactly as CubeEnv does: independent arm
        # initialization, objects at goal, then two random stabilization steps.
        goal_data = self._data_template
        goal_data = self._initialize_arm(goal_data, goal_arm_key)
        goal_cube_qpos = jnp.concatenate(
            [goal_xyzs, jnp.tile(jnp.asarray([[1.0, 0.0, 0.0, 0.0]]), (self.num_cubes, 1))], axis=-1
        )
        goal_data = goal_data.replace(
            qpos=goal_data.qpos.at[self._cube_qpos_ids[:, None] + self._cube_qpos_offsets].set(
                goal_cube_qpos
            ),
            mocap_pos=goal_data.mocap_pos.at[self._cube_target_mocap_ids].set(goal_xyzs),
            mocap_quat=goal_data.mocap_quat.at[self._cube_target_mocap_ids].set(
                jnp.tile(jnp.asarray([[1.0, 0.0, 0.0, 0.0]]), (self.num_cubes, 1))
            ),
        )
        goal_data = mjx.forward(self._mjx_model, goal_data)
        goal_actions = jax.random.uniform(goal_action_key, (2, 5), minval=-1.0, maxval=1.0)
        goal_data = jax.lax.fori_loop(
            0,
            2,
            lambda index, carry: self._physics_step(carry, goal_actions[index]),
            goal_data,
        )

        # Actual task reset: an independent random arm pose plus the task's
        # initial cube positions, 1cm xy jitter, and random cube yaw.
        data = self._initialize_arm(self._data_template, arm_key)
        jitter_key, yaw_key = jax.random.split(cube_key)
        jitter = jax.random.uniform(jitter_key, (self.num_cubes, 2), minval=-0.01, maxval=0.01)
        cube_xyzs = init_xyzs.at[:, :2].add(jitter)
        cube_yaws = jax.random.uniform(yaw_key, (self.num_cubes,), minval=0.0, maxval=2.0 * jnp.pi)
        cube_quats = jax.vmap(_yaw_quat)(cube_yaws)
        cube_qpos = jnp.concatenate([cube_xyzs, cube_quats], axis=-1)
        data = data.replace(
            qpos=data.qpos.at[self._cube_qpos_ids[:, None] + self._cube_qpos_offsets].set(
                cube_qpos
            ),
            mocap_pos=data.mocap_pos.at[self._cube_target_mocap_ids].set(goal_xyzs),
            mocap_quat=data.mocap_quat.at[self._cube_target_mocap_ids].set(
                jnp.tile(jnp.asarray([[1.0, 0.0, 0.0, 0.0]]), (self.num_cubes, 1))
            ),
        )
        data = mjx.forward(self._mjx_model, data)
        return data, goal_data, task_id

    def _reset_impl(self, key: jax.Array, task_ids: jax.Array) -> tuple[jax.Array, CubeJaxState]:
        keys = jax.random.split(key, self.num_envs)
        data, goal_data, task_id = jax.vmap(self._reset_one)(keys, task_ids)
        observation, data = self._observe_batch(data)
        if self.use_oracle_rep:
            goal = jax.vmap(self._oracle_observation)(goal_data)
        elif self.observation_type == "states":
            goal = jax.vmap(self._state_observation)(goal_data)
        else:
            goal, _ = self._observe_batch(goal_data)
        state = CubeJaxState(
            data=data,
            goal=goal,
            task_id=task_id,
            step_count=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            success=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
        )
        return observation, state

    def _step_impl(
        self,
        key: jax.Array,
        state: CubeJaxState,
        action: jax.Array,
        params: CubeJaxParams,
    ) -> tuple[jax.Array, CubeJaxState, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key  # Reserved for Gymnax-compatible stochastic transition logic.
        pre_cube_success = jax.vmap(self._cube_success, in_axes=(0, None))(
            state.data, params.success_threshold
        )
        data = jax.vmap(self._physics_step)(state.data, action)
        post_cube_success = jax.vmap(self._cube_success, in_axes=(0, None))(
            data, params.success_threshold
        )
        post_success = jnp.all(post_cube_success, axis=-1)

        cube_success = pre_cube_success if self.success_timing == "pre" else post_cube_success
        success = jnp.all(cube_success, axis=-1)

        if self.reward_task_id is None:
            reward = success.astype(jnp.float32)
        else:
            reward = (
                jnp.sum(cube_success, axis=-1, dtype=jnp.int32).astype(jnp.float32) - self.num_cubes
            )

        step_count = state.step_count + 1
        terminated = success & self.terminate_at_goal
        truncated = step_count >= params.max_episode_steps
        done = terminated | truncated
        observation, data = self._observe_batch(data)
        next_state = CubeJaxState(
            data=data,
            goal=state.goal,
            task_id=state.task_id,
            step_count=step_count,
            success=post_success,
        )
        info = {
            "success": success,
            "cube_success": cube_success,
            "terminated": terminated,
            "truncated": truncated,
            "task_id": state.task_id,
        }
        return observation, next_state, reward, done, info

    def _rollout_impl(
        self,
        key: jax.Array,
        state: CubeJaxState,
        actions: jax.Array,
        params: CubeJaxParams,
    ) -> tuple[CubeJaxState, CubeJaxTransition]:
        step_keys = jax.random.split(key, actions.shape[0])

        def scan_step(carry, inputs):
            step_key, action = inputs
            observation, next_state, reward, done, info = self._step_impl(
                step_key, carry, action, params
            )
            transition = CubeJaxTransition(observation, reward, done, info)
            return next_state, transition

        return jax.lax.scan(scan_step, state, (step_keys, actions))


def make_cube_jax(*args, **kwargs) -> OGBenchCubeJaxEnv:
    """Create the ``ogbench-cube-jax`` batched functional environment."""

    return OGBenchCubeJaxEnv(*args, **kwargs)


__all__ = (
    "CubeJaxParams",
    "CubeJaxState",
    "CubeJaxTransition",
    "OGBenchCubeJaxEnv",
    "make_cube_jax",
)
