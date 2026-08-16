"""Gymnax-compatible pure-JAX PushT environment."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from envx.pusht.geometry import (
    WINDOW_SIZE,
    coverage,
    diffusion_policy_keypoints,
    origin_from_center_of_mass,
    tee_rectangles,
)
from envx.pusht.physics import simulate_control_step
from envx.pusht.rendering import render as render_state

Array = jax.Array
ObservationType = Literal[
    "state",
    "keypoints",
    "environment_state_agent_pos",
    "pixels",
    "pixels_agent_pos",
]


@struct.dataclass
class EnvState(environment.EnvState):
    """Complete functional simulator state.

    ``block_pos`` is the same body-origin position exposed by the public PushT
    observation. ``block_vel`` is the velocity of the physical center of mass.
    """

    agent_pos: Array
    agent_vel: Array
    block_pos: Array
    block_angle: Array
    block_vel: Array
    block_angular_vel: Array
    last_action: Array
    n_contacts: Array
    time: Array


@struct.dataclass
class EnvParams(environment.EnvParams):
    """PushT task, controller, and solver parameters."""

    dt: float = 0.01
    n_substeps: int = 10
    k_p: float = 100.0
    k_v: float = 20.0
    agent_radius: float = 15.0
    block_inertia: float = 3_000.0
    damping: float = 0.0
    position_correction: float = 0.05
    collision_slop: float = 0.10
    wall_min: float = 7.0
    wall_max: float = 504.0
    goal_x: float = 256.0
    goal_y: float = 256.0
    goal_angle: float = jnp.pi / 4.0
    success_threshold: float = 0.95
    legacy_reset: bool = True
    max_steps_in_episode: int = 300


class PushTEnv(environment.Environment[EnvState, EnvParams]):
    """Massively vectorizable PushT with the Gymnax functional API.

    Args:
        observation_type: ``"state"`` gives the canonical 5-D observation.
            ``"keypoints"`` gives the 20-D Diffusion Policy observation.
            Pixel and Hugging Face dictionary observation variants are also
            available and remain pure JAX.
        observation_size: Height and width of pixel observations.
    """

    _OBSERVATION_TYPES = {
        "state",
        "keypoints",
        "environment_state_agent_pos",
        "pixels",
        "pixels_agent_pos",
    }

    def __init__(
        self,
        observation_type: ObservationType = "state",
        observation_size: int = 96,
    ):
        super().__init__()
        if observation_type not in self._OBSERVATION_TYPES:
            allowed = ", ".join(sorted(self._OBSERVATION_TYPES))
            raise ValueError(f"Unknown observation_type={observation_type!r}; expected {allowed}")
        if observation_size <= 0:
            raise ValueError("observation_size must be positive")
        self.observation_type = observation_type
        self.observation_size = observation_size

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "PushT-JAX-v0"

    @property
    def num_actions(self) -> int:
        return 2

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: Array,
        state: EnvState,
        action: Array,
        params: EnvParams | None = None,
    ) -> tuple[Any, EnvState, Array, Array, dict[str, Array]]:
        """Gymnax transition with its standard automatic reset behavior."""
        if params is None:
            params = self.default_params
        step_key, reset_key = jax.random.split(key)
        observation, next_state, reward, done, info = self.step_env(step_key, state, action, params)
        reset_observation, reset_state = self.reset_env(reset_key, params)
        selected_state = jax.tree.map(
            lambda reset_leaf, step_leaf: jax.lax.select(done, reset_leaf, step_leaf),
            reset_state,
            next_state,
        )
        selected_observation = jax.tree.map(
            lambda reset_leaf, step_leaf: jax.lax.select(done, reset_leaf, step_leaf),
            reset_observation,
            observation,
        )
        return selected_observation, selected_state, reward, done, info

    def reset_env(self, key: Array, params: EnvParams) -> tuple[Any, EnvState]:
        agent_key, block_key, angle_key = jax.random.split(key, 3)
        agent_position = jax.random.randint(agent_key, (2,), 50, 450, dtype=jnp.int32).astype(
            jnp.float32
        )
        raw_block_position = jax.random.randint(block_key, (2,), 100, 400, dtype=jnp.int32).astype(
            jnp.float32
        )
        block_angle = jax.random.uniform(
            angle_key, (), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32
        )

        # The broadly distributed PushT environments set position and then
        # angle. With an off-origin center of gravity this preserves the COM and
        # shifts the observable body origin. Keep that reset behavior by default.
        initial_center = raw_block_position + jnp.asarray((0.0, 45.0), jnp.float32)
        legacy_position = origin_from_center_of_mass(initial_center, block_angle)
        block_position = jnp.where(params.legacy_reset, legacy_position, raw_block_position)

        state = EnvState(
            agent_pos=agent_position,
            agent_vel=jnp.zeros(2, dtype=jnp.float32),
            block_pos=block_position,
            block_angle=block_angle,
            block_vel=jnp.zeros(2, dtype=jnp.float32),
            block_angular_vel=jnp.asarray(0.0, dtype=jnp.float32),
            last_action=agent_position,
            n_contacts=jnp.asarray(0, dtype=jnp.int32),
            time=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state, params), state

    def reset_from_state(
        self, dataset_state: Array, params: EnvParams | None = None
    ) -> tuple[Any, EnvState]:
        """Reset exactly from ``[agent_x, agent_y, block_x, block_y, angle]``.

        Unlike legacy random reset placement, the supplied block position is
        interpreted as the already-observed body origin. This is the useful
        convention for rows loaded from public PushT trajectory datasets.
        """
        if params is None:
            params = self.default_params
        from envx.pusht.trajectory import reset_from_dataset_state

        state = reset_from_dataset_state(dataset_state)
        return self.get_obs(state, params), state

    def step_env(
        self,
        key: Array,
        state: EnvState,
        action: Array,
        params: EnvParams,
    ) -> tuple[Any, EnvState, Array, Array, dict[str, Array]]:
        del key
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), 0.0, WINDOW_SIZE)
        next_state = simulate_control_step(state, action, params).replace(
            last_action=action,
            time=state.time + jnp.asarray(1, dtype=jnp.int32),
        )
        current_coverage = self._coverage(next_state, params)
        reward = jnp.clip(current_coverage / params.success_threshold, 0.0, 1.0)
        terminated = current_coverage > params.success_threshold
        truncated = next_state.time >= params.max_steps_in_episode
        done = terminated | truncated
        observation = self.get_obs(next_state, params)
        info = {
            "block_pose": jnp.concatenate((next_state.block_pos, next_state.block_angle[None])),
            "coverage": current_coverage,
            "discount": jnp.where(done, 0.0, 1.0),
            "goal_pose": jnp.asarray(
                (params.goal_x, params.goal_y, params.goal_angle), dtype=jnp.float32
            ),
            "is_success": terminated,
            "n_contacts": next_state.n_contacts,
            "pos_agent": next_state.agent_pos,
            "terminated": terminated,
            "truncated": truncated,
            "vel_agent": next_state.agent_vel,
        }
        return observation, next_state, reward, done, info

    def _coverage(self, state: EnvState, params: EnvParams) -> Array:
        return coverage(
            state.block_pos,
            state.block_angle,
            jnp.asarray((params.goal_x, params.goal_y), dtype=jnp.float32),
            jnp.asarray(params.goal_angle, dtype=jnp.float32),
        )

    def is_terminated(self, state: EnvState, params: EnvParams) -> Array:
        return self._coverage(state, params) > params.success_threshold

    def is_truncated(self, state: EnvState, params: EnvParams) -> Array:
        return state.time >= params.max_steps_in_episode

    def is_terminal(self, state: EnvState, params: EnvParams) -> Array:
        """Gymnax 0.0.9 terminal predicate (success or time limit)."""
        return self.is_terminated(state, params) | self.is_truncated(state, params)

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams | None = None,
        key: Array | None = None,
    ) -> Any:
        del key
        if params is None:
            params = self.default_params
        if self.observation_type == "state":
            return jnp.concatenate(
                (
                    state.agent_pos,
                    state.block_pos,
                    jnp.mod(state.block_angle, 2.0 * jnp.pi)[None],
                )
            ).astype(jnp.float32)
        if self.observation_type == "keypoints":
            return diffusion_policy_keypoints(state.block_pos, state.block_angle, state.agent_pos)
        if self.observation_type == "environment_state_agent_pos":
            return {
                "agent_pos": state.agent_pos,
                "environment_state": tee_rectangles(state.block_pos, state.block_angle).reshape(-1),
            }

        pixels = self.render(state, params)
        if self.observation_type == "pixels":
            return pixels
        return {"agent_pos": state.agent_pos, "pixels": pixels}

    def render(
        self,
        state: EnvState,
        params: EnvParams | None = None,
        *,
        show_action: bool = False,
    ) -> Array:
        """Return a JAX uint8 RGB array; no display or host transfer occurs."""
        if params is None:
            params = self.default_params
        return render_state(
            state,
            params,
            height=self.observation_size,
            width=self.observation_size,
            show_action=show_action,
        )

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        del params
        return spaces.Box(0.0, WINDOW_SIZE, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams | None = None):
        del params
        if self.observation_type == "state":
            low = jnp.zeros(5, dtype=jnp.float32)
            high = jnp.asarray(
                (WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE, 2.0 * jnp.pi),
                dtype=jnp.float32,
            )
            return spaces.Box(low, high, shape=(5,), dtype=jnp.float32)
        if self.observation_type == "keypoints":
            return spaces.Box(0.0, WINDOW_SIZE, shape=(20,), dtype=jnp.float32)
        if self.observation_type == "environment_state_agent_pos":
            return spaces.Dict(
                {
                    "agent_pos": spaces.Box(0.0, WINDOW_SIZE, shape=(2,), dtype=jnp.float32),
                    "environment_state": spaces.Box(
                        0.0, WINDOW_SIZE, shape=(16,), dtype=jnp.float32
                    ),
                }
            )
        pixel_space = spaces.Box(
            0,
            255,
            shape=(self.observation_size, self.observation_size, 3),
            dtype=jnp.uint8,
        )
        if self.observation_type == "pixels":
            return pixel_space
        return spaces.Dict(
            {
                "agent_pos": spaces.Box(0.0, WINDOW_SIZE, shape=(2,), dtype=jnp.float32),
                "pixels": pixel_space,
            }
        )

    def state_space(self, params: EnvParams | None = None) -> spaces.Dict:
        if params is None:
            params = self.default_params
        maximum = jnp.finfo(jnp.float32).max

        def vector(size):
            return spaces.Box(-maximum, maximum, (size,), jnp.float32)

        def scalar():
            return spaces.Box(-maximum, maximum, (), jnp.float32)

        return spaces.Dict(
            {
                "agent_pos": vector(2),
                "agent_vel": vector(2),
                "block_pos": vector(2),
                "block_angle": scalar(),
                "block_vel": vector(2),
                "block_angular_vel": scalar(),
                "last_action": vector(2),
                "n_contacts": spaces.Discrete(params.n_substeps * 5 + 1),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )
