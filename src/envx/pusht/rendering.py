"""A small pure-JAX rasterizer for optional PushT pixel observations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from envx.pusht.geometry import WINDOW_SIZE, inverse_transform_points, point_in_tee

Array = jax.Array

_WHITE = (255, 255, 255)
_LIGHT_GREEN = (144, 238, 144)
_LIGHT_GRAY = (211, 211, 211)
_ROYAL_BLUE = (65, 105, 225)
_LIGHT_SLATE_GRAY = (119, 136, 153)
_ACTION_RED = (255, 0, 0)


def _paint(image: Array, mask: Array, color: tuple[int, int, int]) -> Array:
    value = jnp.asarray(color, dtype=jnp.uint8)
    return jnp.where(mask[..., None], value, image)


def render(
    state,
    params,
    *,
    height: int = 96,
    width: int = 96,
    show_action: bool = False,
) -> Array:
    """Rasterize one state to an RGB uint8 image entirely on the JAX device.

    World coordinates follow PushT/Pymunk (positive y points upward), while the
    returned image follows the usual top-left image origin.
    """
    rows, columns = jnp.indices((height, width), dtype=jnp.float32)
    world_x = (columns + 0.5) * WINDOW_SIZE / width
    world_y = WINDOW_SIZE - (rows + 0.5) * WINDOW_SIZE / height
    points = jnp.stack((world_x, world_y), axis=-1)

    image = jnp.full((height, width, 3), jnp.asarray(_WHITE, dtype=jnp.uint8))

    goal_position = jnp.asarray((params.goal_x, params.goal_y), dtype=jnp.float32)
    goal_local = inverse_transform_points(points, goal_position, params.goal_angle)
    image = _paint(image, point_in_tee(goal_local), _LIGHT_GREEN)

    wall_mask = (
        (world_x <= params.wall_min)
        | (world_x >= params.wall_max)
        | (world_y <= params.wall_min)
        | (world_y >= params.wall_max)
    )
    image = _paint(image, wall_mask, _LIGHT_GRAY)

    agent_mask = jnp.sum((points - state.agent_pos) ** 2, axis=-1) <= params.agent_radius**2
    image = _paint(image, agent_mask, _ROYAL_BLUE)

    block_local = inverse_transform_points(points, state.block_pos, state.block_angle)
    image = _paint(image, point_in_tee(block_local), _LIGHT_SLATE_GRAY)

    if show_action:
        horizontal = (jnp.abs(world_y - state.last_action[1]) <= WINDOW_SIZE / height) & (
            jnp.abs(world_x - state.last_action[0]) <= 4.0 * WINDOW_SIZE / width
        )
        vertical = (jnp.abs(world_x - state.last_action[0]) <= WINDOW_SIZE / width) & (
            jnp.abs(world_y - state.last_action[1]) <= 4.0 * WINDOW_SIZE / height
        )
        image = _paint(image, horizontal | vertical, _ACTION_RED)

    return image
