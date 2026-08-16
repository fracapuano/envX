import jax
import jax.numpy as jnp
import numpy as np

from envx.tworooms import (
    EnvParams,
    PLDMParams,
    PLDMTwoRoomsEnv,
    TwoRoomsEnv,
    normalize_pldm_location,
    normalize_pldm_pixels,
    pldm_state_from_trajectory,
    render_pldm_locations,
    rollout,
    state_from_proprio,
    state_from_swm_observation,
    swm_observation,
    unnormalize_pldm_location,
)


def test_swm_observation_round_trip():
    observation = jnp.asarray([60, 100, 180, 180, 112, 49, 112, 90, 0, 0], jnp.float32)
    state = state_from_swm_observation(
        observation,
        EnvParams(num_doors=2),
        door_sizes=jnp.asarray([14.0, 18.0, 1.0]),
    )
    np.testing.assert_allclose(swm_observation(state), observation)
    assert int(state.num_doors) == 2


def test_current_rollout_compiles_as_one_scan():
    env = TwoRoomsEnv()
    params = env.default_params
    state = state_from_proprio([60.0, 100.0], [180.0, 100.0], params)
    actions = jnp.tile(jnp.asarray([0.25, 0.0]), (8, 1))
    run = jax.jit(lambda key, initial: rollout(env, key, initial, actions, params))
    final_state, trajectory = run(jax.random.key(0), state)
    np.testing.assert_allclose(final_state.agent_position, [70.0, 100.0], atol=1e-6)
    assert trajectory.observations.shape == (8, 10)
    assert trajectory.actions.shape == (8, 2)
    assert trajectory.info["pos_agent"].shape == (8, 2)


def test_pldm_dataset_scan_replays_free_motion():
    env = PLDMTwoRoomsEnv()
    params = PLDMParams(collision_noise_std=0.0)
    initial = pldm_state_from_trajectory(
        [10.0, 30.0], wall_x=32.0, door_y=10.0, target_position=[50.0, 30.0]
    )
    actions = jnp.asarray([[1.0, 0.0], [0.5, 0.25], [-0.25, 0.5]], jnp.float32)
    final_state, trajectory = jax.jit(
        lambda state: rollout(env, jax.random.key(0), state, actions, params)
    )(initial)
    np.testing.assert_allclose(final_state.position, [11.25, 30.75], atol=1e-6)
    assert trajectory.observations.shape == (3, 2)


def test_pldm_normalization_and_batched_rendering():
    locations = jnp.asarray([[[10.0, 20.0], [11.0, 21.0]], [[30.0, 40.0], [31.0, 41.0]]])
    normalized = normalize_pldm_location(locations)
    np.testing.assert_allclose(unnormalize_pldm_location(normalized), locations, atol=5e-6)
    rendered = jax.jit(render_pldm_locations)(locations)
    assert rendered.shape == (2, 2, 2, 65, 65)
    assert rendered.dtype == jnp.uint8
    normalized_pixels = normalize_pldm_pixels(rendered)
    assert normalized_pixels.shape == rendered.shape
    assert bool(jnp.all(jnp.isfinite(normalized_pixels)))


def test_pldm_batched_rendering_accepts_per_trajectory_layouts():
    locations = jnp.asarray([[[20.0, 20.0], [21.0, 20.0]], [[40.0, 40.0], [41.0, 40.0]]])
    pixels = jax.jit(render_pldm_locations)(
        locations, jnp.asarray([30.0, 35.0]), jnp.asarray([10.0, 20.0])
    )
    assert pixels.shape == (2, 2, 2, 65, 65)
    assert int(pixels[0, 0, 1, 30, 30]) == 255
    assert int(pixels[1, 0, 1, 30, 35]) == 255
