"""Fixed-shape collision kernels for the current TwoRoom-v1 task."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def in_any_door(
    coordinate: Array,
    door_positions: Array,
    door_sizes: Array,
    num_doors: Array,
    margin: float = 1.75,
) -> Array:
    """Return whether a coordinate lies in any active doorway.

    Door sizes follow ``swm/TwoRoom-v1`` and are half-extents, not full widths.
    Arrays always have three entries so the function has a static shape under JIT.
    """
    active = jnp.arange(door_positions.shape[0]) < num_doors
    inside = (coordinate >= door_positions - door_sizes - margin) & (
        coordinate <= door_positions + door_sizes + margin
    )
    return jnp.any(active & inside)


def apply_collisions(
    position: Array,
    proposed_position: Array,
    *,
    agent_radius: Array,
    wall_axis: Array,
    wall_thickness: Array,
    door_positions: Array,
    door_sizes: Array,
    num_doors: Array,
    image_size: float = 224.0,
    border_size: float = 14.0,
    wall_center: float = 112.0,
    door_margin: float = 1.75,
) -> tuple[Array, Array]:
    """Apply the deterministic collision rule used by current TwoRoom-v1.

    ``wall_axis=1`` means a vertical divider and ``wall_axis=0`` a horizontal
    divider. The upstream rule tests the proposed destination against the door,
    then clamps the circle just outside the central wall when it cannot pass.
    """
    lower = border_size + agent_radius
    upper = image_size - border_size - agent_radius
    bounded = jnp.clip(proposed_position, lower, upper)

    half_width = jnp.floor_divide(wall_thickness.astype(jnp.int32), 2).astype(jnp.float32)
    effective_low = wall_center - half_width - agent_radius
    effective_high = wall_center + half_width + agent_radius

    axis = jnp.where(wall_axis == 1, 0, 1)
    along = 1 - axis
    started_low = position[axis] < wall_center
    entered_from_low = started_low & (bounded[axis] > effective_low)
    entered_from_high = (~started_low) & (bounded[axis] < effective_high)
    doorway = in_any_door(bounded[along], door_positions, door_sizes, num_doors, margin=door_margin)
    collided_with_divider = (entered_from_low | entered_from_high) & (~doorway)
    clamped_coordinate = jnp.where(
        started_low,
        effective_low - jnp.asarray(0.5, jnp.float32),
        effective_high + jnp.asarray(0.5, jnp.float32),
    )
    resolved = bounded.at[axis].set(
        jnp.where(collided_with_divider, clamped_coordinate, bounded[axis])
    )
    collided = jnp.any(bounded != proposed_position) | collided_with_divider
    return resolved.astype(jnp.float32), collided
