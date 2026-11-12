"""Reacher and Cube checkerboard setup for the stock MJX-Warp renderer."""

from __future__ import annotations

import copy

import mujoco
import numpy as np


def grid_render_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Match native UV scale, direction, and origin on the two grid floors."""
    rendered = copy.copy(model)
    rendered.material("grid").texrepeat[:] *= (0.5, -0.5)
    grid = rendered.texture("grid")
    height, width = int(grid.height[0]), int(grid.width[0])
    pixels = grid.data.reshape(height, width, 3)
    pixels[:] = np.roll(pixels, (height // 2, width // 2), axis=(0, 1))
    return rendered


def create_grid_render_context(model: mujoco.MjModel, **kwargs):
    """Create the stock context with repeat addressing for its grid texture."""
    import warp as wp
    from mujoco import mjx

    context = mjx.create_render_context(model, **kwargs)
    texture_id = model.texture("grid").id
    for stock in context._contexts.values():
        textures = stock.textures_registry
        texture = textures[texture_id]
        wrapped = wp.Texture2D(
            width=texture.width,
            height=texture.height,
            num_channels=texture.num_channels,
            dtype=texture.dtype,
            address_mode=wp.TextureAddressMode.WRAP,
            device=texture.device,
        )
        wrapped.copy_from(texture)
        # Keep the source alive until its asynchronous creation-time copy finishes.
        stock._envx_grid_source = texture
        textures[texture_id] = wrapped
        stock.textures = wp.array(textures, dtype=wp.Texture2D, device=stock.textures.device)
    return context
