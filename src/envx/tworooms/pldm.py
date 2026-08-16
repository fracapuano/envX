"""Pure-JAX compatibility environment for the classic 65x65 PLDM task."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from envx.tworooms.rendering import render_pldm

Array = jax.Array
PLDMObservationType = Literal["location", "state", "pixels"]


def _vertical_intersection(
    position: Array,
    proposed: Array,
    wall_x: Array,
    hole_y: Array,
    door_space: float,
    has_hole: Array,
) -> tuple[Array, Array]:
    delta = proposed - position
    crosses = jnp.sign(position[0] - wall_x) * jnp.sign(proposed[0] - wall_x) <= 0.1
    non_degenerate = jnp.abs(delta[0]) > 1e-8
    safe_delta_x = jnp.where(non_degenerate, delta[0], 1.0)
    y = position[1] + (wall_x - position[0]) * delta[1] / safe_delta_x
    outside_hole = (y < hole_y - door_space) | (y > hole_y + door_space)
    valid = crosses & non_degenerate & ((~has_hole) | outside_hole)
    return jnp.stack((wall_x, y)), valid


def _horizontal_intersection(
    position: Array,
    proposed: Array,
    wall_y: Array,
) -> tuple[Array, Array]:
    delta = proposed - position
    crosses = jnp.sign(position[1] - wall_y) * jnp.sign(proposed[1] - wall_y) <= 0.1
    non_degenerate = jnp.abs(delta[1]) > 1e-8
    safe_delta_y = jnp.where(non_degenerate, delta[1], 1.0)
    x = position[0] + (wall_y - position[1]) * delta[0] / safe_delta_y
    return jnp.stack((x, wall_y)), crosses & non_degenerate


def resolve_pldm_collision(
    key: Array,
    position: Array,
    proposed: Array,
    wall_x: Array,
    door_y: Array,
    *,
    wall_width: int = 3,
    door_space: float = 4.0,
    border_wall_location: float = 5.0,
    image_size: float = 65.0,
    noise_std: float = 0.5,
) -> tuple[Array, Array]:
    """Port PLDM's segment/wall intersection and rebound distribution."""
    lip_key, vertical_key, horizontal_key = jax.random.split(key, 3)
    left_corner = wall_x - wall_width // 2
    right_corner = wall_x + wall_width // 2
    door_bottom = door_y - door_space
    door_top = door_y + door_space
    movement = proposed - position

    top_intersection, crosses_top = _horizontal_intersection(position, proposed, door_top)
    hits_top_lip = (
        (movement[1] > 0.0)
        & (proposed[1] > door_top)
        & (position[1] < door_top)
        & crosses_top
        & (top_intersection[0] >= left_corner)
        & (top_intersection[0] <= right_corner)
    )
    bottom_intersection, crosses_bottom = _horizontal_intersection(position, proposed, door_bottom)
    hits_bottom_lip = (
        (movement[1] < 0.0)
        & (proposed[1] < door_bottom)
        & (position[1] > door_bottom)
        & crosses_bottom
        & (bottom_intersection[0] >= left_corner)
        & (bottom_intersection[0] <= right_corner)
    )
    lip_intersection = jnp.where(hits_top_lip, top_intersection, bottom_intersection)
    lip_noise = jax.random.normal(lip_key, (2,), dtype=jnp.float32) * noise_std
    lip_noise_y = jnp.where(hits_top_lip, -jnp.abs(lip_noise[1]), jnp.abs(lip_noise[1]))
    lip_result = lip_intersection + lip_noise.at[1].set(lip_noise_y)
    hits_lip = hits_top_lip | hits_bottom_lip

    world_low = border_wall_location - 1.0
    world_high = image_size - border_wall_location
    starts_left = wall_x > position[0]
    left_wall = jnp.where(starts_left, world_low, right_corner)
    right_wall = jnp.where(starts_left, left_corner, world_high)
    left_has_hole = ~starts_left
    right_has_hole = starts_left

    left_intersection, hits_left = _vertical_intersection(
        position, proposed, left_wall, door_y, door_space, left_has_hole
    )
    right_intersection, hits_right = _vertical_intersection(
        position, proposed, right_wall, door_y, door_space, right_has_hole
    )
    vertical_intersection = jnp.where(hits_left, left_intersection, right_intersection)
    hits_vertical = hits_left | hits_right

    top_wall = world_low
    bottom_wall = world_high
    top_border_intersection, hits_top_border = _horizontal_intersection(
        position, proposed, top_wall
    )
    bottom_border_intersection, hits_bottom_border = _horizontal_intersection(
        position, proposed, bottom_wall
    )
    horizontal_intersection = jnp.where(
        hits_top_border, top_border_intersection, bottom_border_intersection
    )
    hits_horizontal = hits_top_border | hits_bottom_border

    vertical_noise = jax.random.normal(vertical_key, (2,), dtype=jnp.float32) * noise_std
    vertical_sign = jnp.sign(position[0] - vertical_intersection[0])
    vertical_noise = vertical_noise.at[0].set(jnp.abs(vertical_noise[0]) * vertical_sign)
    horizontal_noise = jax.random.normal(horizontal_key, (2,), dtype=jnp.float32) * noise_std
    horizontal_sign = jnp.sign(position[1] - horizontal_intersection[1])
    horizontal_noise = horizontal_noise.at[1].set(jnp.abs(horizontal_noise[1]) * horizontal_sign)

    vertical_distance = jnp.linalg.norm(position - vertical_intersection)
    horizontal_distance = jnp.linalg.norm(position - horizontal_intersection)
    choose_vertical = hits_vertical & (
        (~hits_horizontal) | (vertical_distance < horizontal_distance)
    )
    intersection = jnp.where(choose_vertical, vertical_intersection, horizontal_intersection)
    noise = jnp.where(choose_vertical, vertical_noise, horizontal_noise)
    collided = hits_vertical | hits_horizontal
    resolved = intersection + noise
    resolved = jnp.clip(
        resolved, jnp.asarray((left_wall, top_wall)), jnp.asarray((right_wall, bottom_wall))
    )
    resolved = resolved.at[0].set(
        jnp.where(
            resolved[0] <= left_wall,
            left_wall + 0.3,
            jnp.where(resolved[0] >= right_wall, right_wall - 0.3, resolved[0]),
        )
    )
    resolved = resolved.at[1].set(
        jnp.where(
            resolved[1] <= top_wall,
            top_wall + 0.3,
            jnp.where(resolved[1] >= bottom_wall, bottom_wall - 0.3, resolved[1]),
        )
    )
    non_lip_result = jnp.where(collided, resolved, proposed)
    return jnp.where(hits_lip, lip_result, non_lip_result), hits_lip | collided


@struct.dataclass
class PLDMState(environment.EnvState):
    position: Array
    target_position: Array
    wall_x: Array
    door_y: Array
    last_action: Array
    collided: Array
    time: Array


@struct.dataclass
class PLDMParams(environment.EnvParams):
    """Classic PLDM/EB-JEPA Two Rooms parameters."""

    fixed_wall_x: float = 32.0
    fixed_door_y: float = 10.0
    randomize_layout: bool = False
    wall_width: int = 3
    door_space: int = 4
    border_wall_location: int = 5
    collision_noise_std: float = 0.5
    success_mse: float = 1.0
    success_reward: float = 0.0
    max_steps_in_episode: int = 200


class PLDMTwoRoomsEnv(environment.Environment[PLDMState, PLDMParams]):
    """Classic point-and-wall simulator used by PLDM and EB-JEPA trajectories."""

    IMAGE_SIZE = 65

    def __init__(self, observation_type: PLDMObservationType = "location"):
        super().__init__()
        if observation_type not in {"location", "pixels", "state"}:
            raise ValueError("observation_type must be 'location', 'state', or 'pixels'")
        self.observation_type = observation_type

    @property
    def default_params(self) -> PLDMParams:
        return PLDMParams()

    @property
    def name(self) -> str:
        return "PLDMTwoRooms-JAX-v0"

    @property
    def num_actions(self) -> int:
        return 2

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: Array,
        state: PLDMState,
        action: Array,
        params: PLDMParams | None = None,
    ) -> tuple[Any, PLDMState, Array, Array, dict[str, Array]]:
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

    def reset_env(self, key: Array, params: PLDMParams) -> tuple[Any, PLDMState]:
        layout_key, start_key, target_key, direction_key = jax.random.split(key, 4)
        random_wall_x = jax.random.randint(layout_key, (), 20, 45).astype(jnp.float32)
        random_door_y = jax.random.randint(jax.random.fold_in(layout_key, 1), (), 10, 55).astype(
            jnp.float32
        )
        wall_x = jnp.where(params.randomize_layout, random_wall_x, params.fixed_wall_x)
        door_y = jnp.where(params.randomize_layout, random_door_y, params.fixed_door_y)
        padding = jnp.asarray(2.6, jnp.float32)
        border_padding = jnp.asarray(params.border_wall_location - 1, jnp.float32) + padding
        left_wall = wall_x - params.wall_width // 2
        right_wall = wall_x + params.wall_width // 2
        start = jnp.asarray(
            (
                jax.random.uniform(
                    start_key,
                    (),
                    minval=border_padding,
                    maxval=left_wall - padding,
                ),
                jax.random.uniform(
                    jax.random.fold_in(start_key, 1),
                    (),
                    minval=border_padding,
                    maxval=64.0 - border_padding,
                ),
            ),
            jnp.float32,
        )
        target = jnp.asarray(
            (
                jax.random.uniform(
                    target_key,
                    (),
                    minval=right_wall + padding,
                    maxval=64.0 - border_padding,
                ),
                jax.random.uniform(
                    jax.random.fold_in(target_key, 1),
                    (),
                    minval=border_padding,
                    maxval=64.0 - border_padding,
                ),
            ),
            jnp.float32,
        )
        reverse = jax.random.bernoulli(direction_key)
        position = jnp.where(reverse, target, start)
        target_position = jnp.where(reverse, start, target)
        state = PLDMState(
            position=position,
            target_position=target_position,
            wall_x=wall_x.astype(jnp.float32),
            door_y=door_y.astype(jnp.float32),
            last_action=jnp.zeros(2, jnp.float32),
            collided=jnp.asarray(False),
            time=jnp.asarray(0, jnp.int32),
        )
        return self.get_obs(state, params), state

    def reset_from_trajectory(
        self,
        location: Array,
        *,
        wall_x: Array = 32.0,
        door_y: Array = 10.0,
        target_position: Array | None = None,
        params: PLDMParams | None = None,
    ) -> tuple[Any, PLDMState]:
        if params is None:
            params = self.default_params
        if target_position is None:
            target_position = location
        state = pldm_state_from_trajectory(
            location, wall_x=wall_x, door_y=door_y, target_position=target_position
        )
        return self.get_obs(state, params), state

    def step_env(
        self,
        key: Array,
        state: PLDMState,
        action: Array,
        params: PLDMParams,
    ) -> tuple[Any, PLDMState, Array, Array, dict[str, Array]]:
        action = jnp.asarray(action, jnp.float32)
        position, collided = resolve_pldm_collision(
            key,
            state.position,
            state.position + action,
            state.wall_x,
            state.door_y,
            wall_width=params.wall_width,
            door_space=params.door_space,
            border_wall_location=params.border_wall_location,
            noise_std=params.collision_noise_std,
        )
        next_state = state.replace(
            position=position,
            last_action=action,
            collided=collided,
            time=state.time + jnp.asarray(1, jnp.int32),
        )
        mse = jnp.mean((position - state.target_position) ** 2)
        terminated = mse < params.success_mse
        truncated = next_state.time >= params.max_steps_in_episode
        done = terminated | truncated
        reward = terminated.astype(jnp.float32) * params.success_reward
        info = {
            "collided": collided,
            "discount": jnp.where(done, 0.0, 1.0),
            "distance_to_target": jnp.linalg.norm(position - state.target_position),
            "door_y": state.door_y,
            "is_success": terminated,
            "position": position,
            "target_position": state.target_position,
            "terminated": terminated,
            "truncated": truncated,
            "wall_x": state.wall_x,
        }
        return self.get_obs(next_state, params), next_state, reward, done, info

    def get_obs(
        self,
        state: PLDMState,
        params: PLDMParams | None = None,
        key: Array | None = None,
    ) -> Array:
        del key, params
        if self.observation_type == "location":
            return state.position
        if self.observation_type == "state":
            return jnp.concatenate(
                (
                    state.position,
                    state.target_position,
                    state.wall_x[None],
                    state.door_y[None],
                )
            )
        return render_pldm(state.position, state.wall_x, state.door_y)

    def render(self, state: PLDMState, params: PLDMParams | None = None) -> Array:
        del params
        return render_pldm(state.position, state.wall_x, state.door_y)

    def target_pixels(self, state: PLDMState) -> Array:
        return render_pldm(state.target_position, state.wall_x, state.door_y)

    def is_terminated(self, state: PLDMState, params: PLDMParams) -> Array:
        return jnp.mean((state.position - state.target_position) ** 2) < params.success_mse

    def is_truncated(self, state: PLDMState, params: PLDMParams) -> Array:
        return state.time >= params.max_steps_in_episode

    def is_terminal(self, state: PLDMState, params: PLDMParams) -> Array:
        return self.is_terminated(state, params) | self.is_truncated(state, params)

    def action_space(self, params: PLDMParams | None = None) -> spaces.Box:
        del params
        return spaces.Box(-2.45, 2.45, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: PLDMParams | None = None):
        del params
        if self.observation_type == "location":
            return spaces.Box(0.0, 65.0, shape=(2,), dtype=jnp.float32)
        if self.observation_type == "state":
            return spaces.Box(0.0, 65.0, shape=(6,), dtype=jnp.float32)
        return spaces.Box(0, 255, shape=(2, 65, 65), dtype=jnp.uint8)

    def state_space(self, params: PLDMParams | None = None) -> spaces.Dict:
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "position": spaces.Box(0.0, 65.0, (2,), jnp.float32),
                "target_position": spaces.Box(0.0, 65.0, (2,), jnp.float32),
                "wall_x": spaces.Box(0.0, 65.0, (), jnp.float32),
                "door_y": spaces.Box(0.0, 65.0, (), jnp.float32),
                "last_action": self.action_space(params),
                "collided": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )


def pldm_state_from_trajectory(
    location: Array,
    *,
    wall_x: Array = 32.0,
    door_y: Array = 10.0,
    target_position: Array | None = None,
) -> PLDMState:
    """Create state from PLDM/EB-JEPA ``locations/wall_x/door_y`` columns."""
    location = jnp.asarray(location, jnp.float32)
    if target_position is None:
        target_position = location
    return PLDMState(
        position=location,
        target_position=jnp.asarray(target_position, jnp.float32),
        wall_x=jnp.asarray(wall_x, jnp.float32),
        door_y=jnp.asarray(door_y, jnp.float32),
        last_action=jnp.zeros(2, jnp.float32),
        collided=jnp.asarray(False),
        time=jnp.asarray(0, jnp.int32),
    )
