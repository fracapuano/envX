"""Small pure-JAX baseline policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def random_action(key: Array) -> Array:
    """Uniform action for current TwoRoom-v1."""
    return jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0, dtype=jnp.float32)


def weak_expert_action(
    state, *, door_fit_margin: float = 1.1, reach_tolerance: float = 10.5
) -> Array:
    """JAX version of Stable World Model's waypoint-based weak expert."""
    room_axis = jnp.where(state.wall_axis == 1, 0, 1)
    other_axis = 1 - room_axis
    target_other_room = (state.agent_position[room_axis] > 112.0) != (
        state.target_position[room_axis] > 112.0
    )
    active = jnp.arange(3) < state.num_doors
    fits = state.door_sizes >= door_fit_margin * state.agent_radius
    door_points = jnp.zeros((3, 2), jnp.float32)
    door_points = door_points.at[:, room_axis].set(112.0)
    door_points = door_points.at[:, other_axis].set(state.door_positions)
    distances = jnp.linalg.norm(door_points - state.agent_position, axis=-1)
    distances = jnp.where(active & fits, distances, jnp.inf)
    best_index = jnp.argmin(distances)
    best_door = door_points[best_index]
    any_fitting = jnp.any(active & fits)
    fallback = state.target_position.at[room_axis].set(112.0)
    door_waypoint = jnp.where(any_fitting, best_door, fallback)
    waypoint = jnp.where(
        target_other_room & (jnp.linalg.norm(best_door - state.agent_position) > reach_tolerance),
        door_waypoint,
        state.target_position,
    )
    direction = waypoint - state.agent_position
    norm = jnp.linalg.norm(direction)
    return jnp.where(norm > 1e-8, direction / norm, jnp.zeros(2, jnp.float32))


def pldm_random_action(key: Array, max_norm: float = 1.8) -> Array:
    """Isotropic random action suitable for the classic PLDM task."""
    direction_key, magnitude_key = jax.random.split(key)
    direction = jax.random.normal(direction_key, (2,), dtype=jnp.float32)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-8)
    magnitude = jax.random.uniform(magnitude_key, (), minval=0.2, maxval=max_norm)
    return direction * magnitude
