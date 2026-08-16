"""Gymnax-compatible pure-JAX port of current ``swm/TwoRoom-v1``."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from envx.tworooms.geometry import apply_collisions
from envx.tworooms.rendering import render_swm

Array = jax.Array
ObservationType = Literal["state", "pixels", "state_pixels"]


@struct.dataclass
class EnvState(environment.EnvState):
    """Complete per-environment state, including randomized geometry."""

    agent_position: Array
    target_position: Array
    door_positions: Array
    door_sizes: Array
    agent_radius: Array
    target_radius: Array
    speed: Array
    wall_axis: Array
    wall_thickness: Array
    num_doors: Array
    last_action: Array
    time: Array


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Task constants and reset distribution for ``swm/TwoRoom-v1``.

    The upstream defaults randomize agent and target positions at every reset
    but keep all other variations fixed. Set ``randomize_layout=True`` to also
    sample radii, speed, wall axis/thickness, and up to three doors.
    """

    agent_radius: float = 7.0
    target_radius: float = 7.0
    speed: float = 5.0
    wall_axis: int = 1
    wall_thickness: int = 10
    num_doors: int = 1
    door_position_0: float = 49.0
    door_position_1: float = 49.0
    door_position_2: float = 49.0
    door_size_0: float = 14.0
    door_size_1: float = 14.0
    door_size_2: float = 14.0
    success_distance: float = 16.0
    max_steps_in_episode: int = 100
    randomize_layout: bool = False
    force_opposite_rooms: bool = False


def _sample_free_position(
    key: Array,
    wall_axis: Array,
    wall_thickness: Array,
    agent_radius: Array,
) -> Array:
    """Sample uniformly from the upstream position box, rejecting the wall."""
    candidates = jax.random.uniform(
        key,
        (32, 2),
        minval=jnp.asarray(14.0, jnp.float32),
        maxval=jnp.asarray(209.0, jnp.float32),
    )
    axis = jnp.where(wall_axis == 1, 0, 1)
    half_width = jnp.floor_divide(wall_thickness, 2).astype(jnp.float32)
    wall_min = 112.0 - half_width - agent_radius
    wall_max = 112.0 + half_width + agent_radius
    valid = (candidates[:, axis] < wall_min) | (candidates[:, axis] > wall_max)
    index = jnp.argmax(valid.astype(jnp.int32))
    return candidates[index]


class TwoRoomsEnv(environment.Environment[EnvState, EnvParams]):
    """Massively vectorizable current TwoRoom-v1 environment.

    Args:
        observation_type: ``"state"`` is the upstream 10-D observation and is
            fastest. ``"pixels"`` is a JAX-rendered 224x224 HWC uint8 frame.
            ``"state_pixels"`` returns both in a dictionary.
        visualize_goal: Draw the green goal in pixel observations. This is
            false by default to match datasets that supply goal state without
            drawing it in agent observations. It does not change task logic.
    """

    _OBSERVATION_TYPES = {"pixels", "state", "state_pixels"}
    IMAGE_SIZE = 224
    MAX_DOORS = 3

    def __init__(
        self,
        observation_type: ObservationType = "state",
        *,
        visualize_goal: bool = False,
    ):
        super().__init__()
        if observation_type not in self._OBSERVATION_TYPES:
            allowed = ", ".join(sorted(self._OBSERVATION_TYPES))
            raise ValueError(f"Unknown observation_type={observation_type!r}; expected {allowed}")
        self.observation_type = observation_type
        self.visualize_goal = bool(visualize_goal)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "TwoRooms-JAX-v1"

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
        """Gymnax step with standard automatic reset after terminal states."""
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
        keys = jax.random.split(key, 10)
        random_agent_radius = jax.random.uniform(keys[0], (), minval=7.0, maxval=14.0)
        random_target_radius = jax.random.uniform(keys[1], (), minval=7.0, maxval=14.0)
        random_speed = jax.random.uniform(keys[2], (), minval=1.75, maxval=10.5)
        random_axis = jax.random.randint(keys[3], (), 0, 2, dtype=jnp.int32)
        random_thickness = jax.random.randint(keys[4], (), 7, 42, dtype=jnp.int32)
        random_num_doors = jax.random.randint(keys[5], (), 1, 4, dtype=jnp.int32)
        random_door_positions = jax.random.randint(keys[6], (3,), 0, 224).astype(jnp.float32)
        random_door_sizes = jax.random.randint(keys[7], (3,), 1, 22).astype(jnp.float32)

        randomize = jnp.asarray(params.randomize_layout)
        agent_radius = jnp.where(randomize, random_agent_radius, params.agent_radius).astype(
            jnp.float32
        )
        target_radius = jnp.where(randomize, random_target_radius, params.target_radius).astype(
            jnp.float32
        )
        speed = jnp.where(randomize, random_speed, params.speed).astype(jnp.float32)
        wall_axis = jnp.where(randomize, random_axis, params.wall_axis).astype(jnp.int32)
        wall_thickness = jnp.where(randomize, random_thickness, params.wall_thickness).astype(
            jnp.int32
        )
        num_doors = jnp.where(randomize, random_num_doors, params.num_doors).astype(jnp.int32)
        fixed_door_positions = jnp.asarray(
            (params.door_position_0, params.door_position_1, params.door_position_2),
            dtype=jnp.float32,
        )
        fixed_door_sizes = jnp.asarray(
            (params.door_size_0, params.door_size_1, params.door_size_2),
            dtype=jnp.float32,
        )
        door_positions = jnp.where(randomize, random_door_positions, fixed_door_positions)
        door_sizes = jnp.where(randomize, random_door_sizes, fixed_door_sizes)
        minimum_fitting_size = jnp.ceil(1.1 * agent_radius)
        door_sizes = door_sizes.at[0].set(
            jnp.where(randomize, jnp.maximum(door_sizes[0], minimum_fitting_size), door_sizes[0])
        )

        agent_position = _sample_free_position(keys[8], wall_axis, wall_thickness, agent_radius)
        target_position = jax.random.uniform(
            keys[9], (2,), minval=14.0, maxval=209.0, dtype=jnp.float32
        )
        room_axis = jnp.where(wall_axis == 1, 0, 1)
        same_room = (agent_position[room_axis] < 112.0) == (target_position[room_axis] < 112.0)
        target_position = target_position.at[room_axis].set(
            jnp.where(
                params.force_opposite_rooms & same_room,
                224.0 - target_position[room_axis],
                target_position[room_axis],
            )
        )

        state = EnvState(
            agent_position=agent_position,
            target_position=target_position,
            door_positions=door_positions,
            door_sizes=door_sizes,
            agent_radius=agent_radius,
            target_radius=target_radius,
            speed=speed,
            wall_axis=wall_axis,
            wall_thickness=wall_thickness,
            num_doors=num_doors,
            last_action=jnp.zeros(2, dtype=jnp.float32),
            time=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state, params), state

    def reset_from_state(
        self,
        agent_position: Array,
        target_position: Array,
        params: EnvParams | None = None,
        *,
        door_positions: Array | None = None,
        door_sizes: Array | None = None,
        wall_axis: int | Array | None = None,
    ) -> tuple[Any, EnvState]:
        """Construct an episode from stored proprio/goal and optional layout."""
        if params is None:
            params = self.default_params
        state = state_from_proprio(
            agent_position,
            target_position,
            params,
            door_positions=door_positions,
            door_sizes=door_sizes,
            wall_axis=wall_axis,
        )
        return self.get_obs(state, params), state

    def step_env(
        self,
        key: Array,
        state: EnvState,
        action: Array,
        params: EnvParams,
    ) -> tuple[Any, EnvState, Array, Array, dict[str, Array]]:
        del key
        clipped_action = jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)
        proposed_position = state.agent_position + clipped_action * state.speed
        agent_position, collided = apply_collisions(
            state.agent_position,
            proposed_position,
            agent_radius=state.agent_radius,
            wall_axis=state.wall_axis,
            wall_thickness=state.wall_thickness,
            door_positions=state.door_positions,
            door_sizes=state.door_sizes,
            num_doors=state.num_doors,
        )
        next_state = state.replace(
            agent_position=agent_position,
            last_action=clipped_action,
            time=state.time + jnp.asarray(1, jnp.int32),
        )
        distance = jnp.linalg.norm(agent_position - state.target_position)
        terminated = distance < params.success_distance
        truncated = next_state.time >= params.max_steps_in_episode
        done = terminated | truncated
        reward = jnp.asarray(0.0, jnp.float32)
        info = {
            "collided": collided,
            "discount": jnp.where(done, 0.0, 1.0),
            "distance_to_target": distance,
            "goal_state": state.target_position,
            "is_success": terminated,
            "pos_agent": agent_position,
            "proprio": agent_position,
            "terminated": terminated,
            "truncated": truncated,
        }
        return self.get_obs(next_state, params), next_state, reward, done, info

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams | None = None,
        key: Array | None = None,
    ) -> Any:
        del key, params
        door_coordinate = jnp.where(
            state.wall_axis == 1,
            jnp.stack((jnp.full(3, 112.0), state.door_positions), axis=-1),
            jnp.stack((state.door_positions, jnp.full(3, 112.0)), axis=-1),
        )
        active = (jnp.arange(3) < state.num_doors)[:, None]
        door_coordinate = jnp.where(active, door_coordinate, 0.0).reshape(-1)
        state_observation = jnp.concatenate(
            (state.agent_position, state.target_position, door_coordinate)
        ).astype(jnp.float32)
        if self.observation_type == "state":
            return state_observation
        pixels = self.render(state)
        if self.observation_type == "pixels":
            return pixels
        return {"pixels": pixels, "state": state_observation}

    def render(self, state: EnvState, params: EnvParams | None = None) -> Array:
        del params
        return render_swm(state, visualize_goal=self.visualize_goal)

    def target_pixels(self, state: EnvState) -> Array:
        """Return upstream-style target pixels (red agent placed at goal)."""
        return render_swm(state, visualize_goal=False, agent_position=state.target_position)

    def is_terminated(self, state: EnvState, params: EnvParams) -> Array:
        return (
            jnp.linalg.norm(state.agent_position - state.target_position) < params.success_distance
        )

    def is_truncated(self, state: EnvState, params: EnvParams) -> Array:
        return state.time >= params.max_steps_in_episode

    def is_terminal(self, state: EnvState, params: EnvParams) -> Array:
        return self.is_terminated(state, params) | self.is_truncated(state, params)

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        del params
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams | None = None):
        del params
        state_space = spaces.Box(0.0, 224.0, shape=(10,), dtype=jnp.float32)
        if self.observation_type == "state":
            return state_space
        pixel_space = spaces.Box(0, 255, shape=(224, 224, 3), dtype=jnp.uint8)
        if self.observation_type == "pixels":
            return pixel_space
        return spaces.Dict({"pixels": pixel_space, "state": state_space})

    def state_space(self, params: EnvParams | None = None) -> spaces.Dict:
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "agent_position": spaces.Box(0.0, 224.0, (2,), jnp.float32),
                "target_position": spaces.Box(0.0, 224.0, (2,), jnp.float32),
                "door_positions": spaces.Box(0.0, 224.0, (3,), jnp.float32),
                "door_sizes": spaces.Box(1.0, 21.0, (3,), jnp.float32),
                "agent_radius": spaces.Box(7.0, 14.0, (), jnp.float32),
                "target_radius": spaces.Box(7.0, 14.0, (), jnp.float32),
                "speed": spaces.Box(1.75, 10.5, (), jnp.float32),
                "wall_axis": spaces.Discrete(2),
                "wall_thickness": spaces.Discrete(42),
                "num_doors": spaces.Discrete(4),
                "last_action": self.action_space(params),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )


def state_from_proprio(
    agent_position: Array,
    target_position: Array,
    params: EnvParams | None = None,
    *,
    door_positions: Array | None = None,
    door_sizes: Array | None = None,
    wall_axis: int | Array | None = None,
) -> EnvState:
    """Create simulator state from Stable World Model trajectory columns."""
    if params is None:
        params = EnvParams()
    if door_positions is None:
        door_positions = jnp.asarray(
            (params.door_position_0, params.door_position_1, params.door_position_2),
            jnp.float32,
        )
    if door_sizes is None:
        door_sizes = jnp.asarray(
            (params.door_size_0, params.door_size_1, params.door_size_2), jnp.float32
        )
    if wall_axis is None:
        wall_axis = params.wall_axis
    return EnvState(
        agent_position=jnp.asarray(agent_position, jnp.float32),
        target_position=jnp.asarray(target_position, jnp.float32),
        door_positions=jnp.asarray(door_positions, jnp.float32),
        door_sizes=jnp.asarray(door_sizes, jnp.float32),
        agent_radius=jnp.asarray(params.agent_radius, jnp.float32),
        target_radius=jnp.asarray(params.target_radius, jnp.float32),
        speed=jnp.asarray(params.speed, jnp.float32),
        wall_axis=jnp.asarray(wall_axis, jnp.int32),
        wall_thickness=jnp.asarray(params.wall_thickness, jnp.int32),
        num_doors=jnp.asarray(params.num_doors, jnp.int32),
        last_action=jnp.zeros(2, jnp.float32),
        time=jnp.asarray(0, jnp.int32),
    )


TwoRooms = TwoRoomsEnv
