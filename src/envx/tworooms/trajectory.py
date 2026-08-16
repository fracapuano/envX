"""Compiled rollout and public trajectory-format adapters."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from envx.tworooms.environment import EnvParams, EnvState, state_from_proprio
from envx.tworooms.pldm import PLDMParams, PLDMState, pldm_state_from_trajectory
from envx.tworooms.rendering import render_pldm

Array = jax.Array

PLDM_LOCATION_MEAN = jnp.asarray((31.5863, 32.0618), jnp.float32)
PLDM_LOCATION_STD = jnp.asarray((16.1025, 16.1353), jnp.float32)
PLDM_PIXEL_MEAN = jnp.asarray((0.0026, 0.0989), jnp.float32)
PLDM_PIXEL_STD = jnp.asarray((0.0369, 0.2986), jnp.float32)


@struct.dataclass
class Trajectory:
    """Static-shape outputs from a JAX ``lax.scan`` rollout."""

    observations: Any
    actions: Array
    rewards: Array
    terminated: Array
    truncated: Array
    info: dict[str, Array]


def rollout(
    env,
    key: Array,
    initial_state,
    actions: Array,
    params=None,
    *,
    auto_reset: bool = False,
):
    """Execute an action sequence in one compiled scan.

    Offline-dataset comparison should use the default ``auto_reset=False`` so
    terminal states remain visible. Set it true for standard Gymnax behavior.
    """
    if params is None:
        params = env.default_params
    keys = jax.random.split(key, actions.shape[0])

    def scan_step(current_state, inputs):
        step_key, action = inputs
        step_function = env.step if auto_reset else env.step_env
        observation, next_state, reward, _done, info = step_function(
            step_key, current_state, action, params
        )
        outputs = (
            observation,
            reward,
            info["terminated"],
            info["truncated"],
            info,
        )
        return next_state, outputs

    final_state, outputs = jax.lax.scan(scan_step, initial_state, (keys, actions))
    observations, rewards, terminated, truncated, info = outputs
    return final_state, Trajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


def state_from_swm_observation(
    observation: Array,
    params: EnvParams | None = None,
    *,
    door_sizes: Array | None = None,
    wall_axis: int | Array = 1,
) -> EnvState:
    """Create state from the current Stable World Model 10-value observation."""
    observation = jnp.asarray(observation, jnp.float32)
    door_coordinates = observation[4:].reshape(3, 2)
    wall_axis_array = jnp.asarray(wall_axis, jnp.int32)
    door_positions = jnp.where(wall_axis_array == 1, door_coordinates[:, 1], door_coordinates[:, 0])
    active = jnp.any(door_coordinates != 0.0, axis=1)
    if params is None:
        params = EnvParams()
    state = state_from_proprio(
        observation[:2],
        observation[2:4],
        params,
        door_positions=door_positions,
        door_sizes=door_sizes,
        wall_axis=wall_axis_array,
    )
    return state.replace(num_doors=jnp.sum(active).astype(jnp.int32))


def swm_observation(state: EnvState) -> Array:
    """Export current state to the public 10-value observation layout."""
    door_coordinates = jnp.where(
        state.wall_axis == 1,
        jnp.stack((jnp.full(3, 112.0), state.door_positions), axis=-1),
        jnp.stack((state.door_positions, jnp.full(3, 112.0)), axis=-1),
    )
    active = (jnp.arange(3) < state.num_doors)[:, None]
    return jnp.concatenate(
        (
            state.agent_position,
            state.target_position,
            jnp.where(active, door_coordinates, 0.0).reshape(-1),
        )
    )


def normalize_pldm_location(location: Array) -> Array:
    """Apply the location statistics distributed with EB-JEPA."""
    return (jnp.asarray(location, jnp.float32) - PLDM_LOCATION_MEAN) / (PLDM_LOCATION_STD + 1e-6)


def unnormalize_pldm_location(location: Array) -> Array:
    return jnp.asarray(location, jnp.float32) * PLDM_LOCATION_STD + PLDM_LOCATION_MEAN


def normalize_pldm_pixels(pixels: Array) -> Array:
    """Match EB-JEPA's per-channel min-max and standard normalization."""
    pixels = jnp.asarray(pixels, jnp.float32)
    minimum = jnp.min(pixels, axis=(-2, -1), keepdims=True)
    scaled = pixels - minimum
    scaled = scaled / (jnp.max(scaled, axis=(-2, -1), keepdims=True) + 1e-6)
    shape = (1,) * (scaled.ndim - 3) + (2, 1, 1)
    return (scaled - PLDM_PIXEL_MEAN.reshape(shape)) / (PLDM_PIXEL_STD.reshape(shape) + 1e-6)


def render_pldm_locations(
    locations: Array,
    wall_x: Array = 32.0,
    door_y: Array = 10.0,
) -> Array:
    """Render ``[..., T, 2]`` trajectory locations with JAX batching."""
    locations = jnp.asarray(locations, jnp.float32)
    wall_x = jnp.asarray(wall_x, jnp.float32)
    door_y = jnp.asarray(door_y, jnp.float32)
    if wall_x.ndim == 0 and door_y.ndim == 0:
        flat = locations.reshape((-1, 2))
        pixels = jax.vmap(lambda location: render_pldm(location, wall_x, door_y))(flat)
        return pixels.reshape(locations.shape[:-1] + (2, 65, 65))
    if locations.ndim < 3:
        raise ValueError("Per-trajectory layouts require locations with shape [batch, time, 2]")
    return jax.vmap(
        lambda trajectory, current_wall_x, current_door_y: jax.vmap(
            lambda location: render_pldm(location, current_wall_x, current_door_y)
        )(trajectory)
    )(locations, wall_x, door_y)


__all__ = [
    "PLDMParams",
    "PLDMState",
    "Trajectory",
    "normalize_pldm_location",
    "normalize_pldm_pixels",
    "pldm_state_from_trajectory",
    "render_pldm_locations",
    "rollout",
    "state_from_swm_observation",
    "swm_observation",
    "unnormalize_pldm_location",
]
