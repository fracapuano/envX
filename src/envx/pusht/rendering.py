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
    if mask.dtype == jnp.bool_:
        return jnp.where(mask[..., None], value, image)
    alpha = mask[..., None]
    return jnp.rint(image * (1 - alpha) + value * alpha).astype(jnp.uint8)


def _light_color(color: tuple[int, int, int]) -> Array:
    # SWM's Pymunk DrawOptions draws lighter interiors inside the body outlines.
    return jnp.rint(jnp.minimum(jnp.asarray(color, jnp.float32) * 1.2, 255)).astype(jnp.uint8)


def _rectangle_masks(points: Array, lower: Array, upper: Array) -> tuple[Array, Array]:
    """Return the two-world-unit outline and interior of a Pymunk polygon."""
    center = (lower + upper) / 2
    distance = jnp.abs(points - center) - (upper - lower) / 2
    outside = jnp.linalg.norm(jnp.maximum(distance, 0), axis=-1)
    inside = jnp.minimum(jnp.max(distance, axis=-1), 0)
    signed_distance = outside + inside
    return signed_distance <= 2, signed_distance < -2


def render(
    state,
    params,
    *,
    height: int = 96,
    width: int = 96,
    show_action: bool = False,
) -> Array:
    """Rasterize one state to an RGB uint8 image entirely on the JAX device.

    Match SWM's Pygame view: x points right and y points down. Draw the same
    body colors, outlines and wall segments, sampling directly at the requested
    resolution. Pygame's integer rasterization/resize can differ at edges.
    """
    rows, columns = jnp.indices((height, width), dtype=jnp.float32)
    world_x = (columns + 0.5) * WINDOW_SIZE / width
    world_y = (rows + 0.5) * WINDOW_SIZE / height
    points = jnp.stack((world_x, world_y), axis=-1)

    image = jnp.full((height, width, 3), jnp.asarray(_WHITE, dtype=jnp.uint8))

    goal_position = jnp.asarray((params.goal_x, params.goal_y), dtype=jnp.float32)
    goal_local = inverse_transform_points(points, goal_position, params.goal_angle)
    image = _paint(image, point_in_tee(goal_local), _LIGHT_GREEN)

    # The collision bounds are the inside wall faces; native rendering draws
    # four radius-two segments, leaving the image margins white.
    wall_low = params.wall_min - 2
    wall_high = params.wall_max + 2

    def wall_coverage(coordinate):
        # Match bilinear sampling of Pygame's five-pixel radius-two segments.
        distance = jnp.minimum(
            jnp.abs(coordinate - wall_low - 0.5), jnp.abs(coordinate - wall_high - 0.5)
        )
        return jnp.clip(3 - distance, 0, 1)

    vertical = wall_coverage(world_x) * ((world_y >= wall_low - 2) & (world_y <= wall_high + 2))
    horizontal = wall_coverage(world_y) * ((world_x >= wall_low - 2) & (world_x <= wall_high + 2))
    wall_mask = jnp.maximum(vertical, horizontal)
    image = _paint(image, wall_mask, _LIGHT_GRAY)

    agent_mask = jnp.sum((points - state.agent_pos) ** 2, axis=-1) <= params.agent_radius**2
    image = _paint(image, agent_mask, _ROYAL_BLUE)
    agent_inner = (
        jnp.sum((points - state.agent_pos) ** 2, axis=-1)
        <= jnp.maximum(params.agent_radius - 4, 0) ** 2
    )
    image = _paint(image, agent_inner, _light_color(_ROYAL_BLUE))

    block_local = inverse_transform_points(points, state.block_pos, state.block_angle)
    # Draw both Pymunk polygons, including the shared edge at the T junction.
    for lower, upper in (((-60, 0), (60, 30)), ((-15, 30), (15, 120))):
        outer, inner = _rectangle_masks(block_local, jnp.asarray(lower), jnp.asarray(upper))
        image = _paint(image, outer, _LIGHT_SLATE_GRAY)
        image = _paint(image, inner, _light_color(_LIGHT_SLATE_GRAY))

    if show_action:
        horizontal = (jnp.abs(world_y - state.last_action[1]) <= WINDOW_SIZE / height) & (
            jnp.abs(world_x - state.last_action[0]) <= 4.0 * WINDOW_SIZE / width
        )
        vertical = (jnp.abs(world_x - state.last_action[0]) <= WINDOW_SIZE / width) & (
            jnp.abs(world_y - state.last_action[1]) <= 4.0 * WINDOW_SIZE / height
        )
        image = _paint(image, horizontal | vertical, _ACTION_RED)

    return image
