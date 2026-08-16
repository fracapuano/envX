"""Pure-JAX rasterizers for both supported Two Rooms definitions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def _alpha_blend(image: Array, alpha: Array, color: tuple[int, int, int]) -> Array:
    alpha = jnp.clip(alpha, 0.0, 1.0)[..., None]
    rgb = jnp.asarray(color, dtype=jnp.float32)
    blended = image.astype(jnp.float32) * (1.0 - alpha) + rgb * alpha
    return jnp.clip(blended, 0.0, 255.0).astype(jnp.uint8)


def _gaussian_dot(position: Array, radius: Array, image_size: int) -> Array:
    coordinates = jnp.arange(image_size, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    distance_squared = (grid_x - position[0]) ** 2 + (grid_y - position[1]) ** 2
    standard_deviation = jnp.maximum(radius, jnp.asarray(1e-6, jnp.float32))
    dot = jnp.exp(-distance_squared / (2.0 * standard_deviation**2))
    return dot / jnp.maximum(jnp.max(dot), jnp.asarray(1e-12, jnp.float32))


def render_swm(
    state,
    *,
    visualize_goal: bool = False,
    agent_position: Array | None = None,
    image_size: int = 224,
) -> Array:
    """Render an HWC uint8 frame matching ``swm/TwoRoom-v1`` defaults."""
    coordinates = jnp.arange(image_size, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    half_width = jnp.floor_divide(state.wall_thickness, 2)

    vertical_stripe = (grid_x >= 112 - half_width) & (grid_x <= 112 + half_width)
    horizontal_stripe = (grid_y >= 112 - half_width) & (grid_y <= 112 + half_width)
    wall_stripe = jnp.where(state.wall_axis == 1, vertical_stripe, horizontal_stripe)

    active = jnp.arange(3) < state.num_doors
    vertical_openings = (
        grid_y[None] >= state.door_positions[:, None, None] - state.door_sizes[:, None, None]
    ) & (grid_y[None] <= state.door_positions[:, None, None] + state.door_sizes[:, None, None])
    horizontal_openings = (
        grid_x[None] >= state.door_positions[:, None, None] - state.door_sizes[:, None, None]
    ) & (grid_x[None] <= state.door_positions[:, None, None] + state.door_sizes[:, None, None])
    openings = jnp.where(state.wall_axis == 1, vertical_openings, horizontal_openings)
    door_span = jnp.any(openings & active[:, None, None], axis=0)
    wall_mask = wall_stripe & (~door_span)

    border = jnp.zeros((image_size, image_size), dtype=jnp.bool_)
    border = border.at[:, 10:14].set(True)
    border = border.at[:, image_size - 14 : image_size - 10].set(True)
    border = border.at[10:14, :].set(True)
    border = border.at[image_size - 14 : image_size - 10, :].set(True)
    wall_mask = wall_mask | border

    image = jnp.full((image_size, image_size, 3), 255, dtype=jnp.uint8)
    image = jnp.where(wall_mask[..., None], jnp.zeros(3, dtype=jnp.uint8), image)
    if visualize_goal:
        target_dot = _gaussian_dot(state.target_position, state.target_radius, image_size)
        image = _alpha_blend(image, target_dot, (0, 255, 0))
    position = state.agent_position if agent_position is None else agent_position
    agent_dot = _gaussian_dot(position, state.agent_radius, image_size)
    return _alpha_blend(image, agent_dot, (255, 0, 0))


def render_pldm_dot(position: Array, dot_std: float = 1.3, image_size: int = 65) -> Array:
    """Render the canonical PLDM Gaussian dot channel as uint8."""
    coordinates = jnp.arange(image_size, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    distance_squared = (grid_x - position[0]) ** 2 + (grid_y - position[1]) ** 2
    values = jnp.exp(-distance_squared / (2.0 * dot_std**2)) * 255.0
    return jnp.clip(values, 0.0, 255.0).astype(jnp.uint8)


def render_pldm_walls(
    wall_x: Array,
    door_y: Array,
    *,
    image_size: int = 65,
    border_wall_location: int = 5,
    wall_width: int = 3,
    door_space: int = 4,
) -> Array:
    """Render the PLDM wall channel as uint8."""
    coordinates = jnp.arange(image_size)
    grid_y, grid_x = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    half_width = wall_width // 2
    central = (grid_x >= wall_x - half_width) & (grid_x <= wall_x + half_width)
    door = (grid_y >= door_y - door_space) & (grid_y <= door_y + door_space)
    walls = central & (~door)
    border_index = border_wall_location - 1
    far_index = image_size - border_wall_location
    walls = walls.at[:, border_index].set(True)
    walls = walls.at[:, far_index].set(True)
    walls = walls.at[border_index, :].set(True)
    walls = walls.at[far_index, :].set(True)
    return (walls.astype(jnp.uint8) * jnp.asarray(255, jnp.uint8)).astype(jnp.uint8)


def render_pldm(position: Array, wall_x: Array, door_y: Array) -> Array:
    """Return canonical PLDM pixels in channel-first ``(2, 65, 65)`` layout."""
    return jnp.stack((render_pldm_dot(position), render_pldm_walls(wall_x, door_y)))
