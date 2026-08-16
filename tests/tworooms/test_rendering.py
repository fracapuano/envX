import jax
import jax.numpy as jnp
import numpy as np

from envx.tworooms import EnvParams, PLDMTwoRoomsEnv, TwoRoomsEnv, state_from_proprio
from envx.tworooms.rendering import render_pldm_dot, render_pldm_walls


def test_current_renderer_geometry_dtype_and_jit():
    env = TwoRoomsEnv(observation_type="pixels")
    state = state_from_proprio(jnp.array([60.0, 100.0]), jnp.array([180.0, 180.0]), EnvParams())
    pixels = jax.jit(env.render)(state)
    assert pixels.shape == (224, 224, 3)
    assert pixels.dtype == jnp.uint8
    np.testing.assert_array_equal(pixels[100, 60], [255, 0, 0])
    np.testing.assert_array_equal(pixels[100, 112], [0, 0, 0])
    np.testing.assert_array_equal(pixels[49, 112], [255, 255, 255])
    np.testing.assert_array_equal(pixels[12, 30], [0, 0, 0])
    np.testing.assert_array_equal(pixels[180, 180], [255, 255, 255])

    target = jax.jit(env.target_pixels)(state)
    np.testing.assert_array_equal(target[180, 180], [255, 0, 0])


def test_visualize_goal_is_opt_in_and_green_beneath_agent():
    env = TwoRoomsEnv(observation_type="pixels", visualize_goal=True)
    state = state_from_proprio(jnp.array([60.0, 100.0]), jnp.array([180.0, 180.0]), EnvParams())
    pixels = env.render(state)
    np.testing.assert_array_equal(pixels[180, 180], [0, 255, 0])


def test_visualize_goal_only_changes_pixels():
    params = EnvParams()
    state = state_from_proprio(jnp.array([60.0, 100.0]), jnp.array([180.0, 180.0]), params)
    hidden = TwoRoomsEnv(observation_type="pixels")
    visible = TwoRoomsEnv(observation_type="pixels", visualize_goal=True)

    assert hidden.visualize_goal is False
    assert visible.visualize_goal is True
    key = jax.random.key(7)
    action = jnp.array([0.5, -0.25], dtype=jnp.float32)
    hidden_output = hidden.step_env(key, state, action, params)
    visible_output = visible.step_env(key, state, action, params)
    hidden_pixels, hidden_state, hidden_reward, hidden_done, hidden_info = hidden_output
    visible_pixels, visible_state, visible_reward, visible_done, visible_info = visible_output

    np.testing.assert_array_equal(hidden_pixels[180, 180], [255, 255, 255])
    np.testing.assert_array_equal(visible_pixels[180, 180], [0, 255, 0])
    for hidden_leaf, visible_leaf in zip(
        jax.tree.leaves(hidden_state), jax.tree.leaves(visible_state), strict=True
    ):
        np.testing.assert_array_equal(hidden_leaf, visible_leaf)
    np.testing.assert_array_equal(hidden_reward, visible_reward)
    np.testing.assert_array_equal(hidden_done, visible_done)
    for name in hidden_info:
        np.testing.assert_array_equal(hidden_info[name], visible_info[name])


def test_pldm_dot_matches_closed_form_to_uint8_rounding():
    position = jnp.array([20.25, 30.75], jnp.float32)
    rendered = np.asarray(render_pldm_dot(position))
    y, x = np.meshgrid(np.arange(65), np.arange(65), indexing="ij")
    expected = np.clip(
        np.exp(-((x - 20.25) ** 2 + (y - 30.75) ** 2) / (2 * 1.3**2)) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    np.testing.assert_allclose(rendered, expected, atol=1)


def test_pldm_wall_pixels_are_canonical():
    walls = np.asarray(render_pldm_walls(jnp.array(32.0), jnp.array(10.0)))
    assert walls.shape == (65, 65)
    assert walls.dtype == np.uint8
    assert walls[30, 32] == 255
    assert walls[10, 32] == 0
    assert walls[4, 20] == 255
    assert walls[60, 20] == 255
    assert walls[20, 4] == 255
    assert walls[20, 60] == 255

    env = PLDMTwoRoomsEnv(observation_type="pixels")
    observation, _ = env.reset(jax.random.key(0), env.default_params)
    assert observation.shape == (2, 65, 65)
    assert observation.dtype == jnp.uint8
