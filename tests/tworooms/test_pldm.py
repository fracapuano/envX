import jax
import jax.numpy as jnp
import numpy as np

from envx.tworooms import PLDMParams, PLDMTwoRoomsEnv, pldm_state_from_trajectory


def _step(position, action, *, door_y=10.0):
    env = PLDMTwoRoomsEnv()
    params = PLDMParams(collision_noise_std=0.0)
    state = pldm_state_from_trajectory(
        position, wall_x=32.0, door_y=door_y, target_position=[55, 55]
    )
    return env.step_env(jax.random.key(0), state, jnp.asarray(action), params)


def test_free_transition_is_exact_addition():
    _, state, _, done, info = _step([10.0, 30.0], [1.25, -0.5])
    np.testing.assert_allclose(state.position, [11.25, 29.5], atol=0.0)
    assert not bool(done)
    assert not bool(info["collided"])


def test_central_wall_border_and_door_fixtures():
    _, state, _, _, info = _step([29.0, 30.0], [3.0, 0.0])
    np.testing.assert_allclose(state.position, [30.7, 30.0], atol=1e-6)
    assert bool(info["collided"])

    _, state, _, _, info = _step([29.0, 10.0], [5.0, 0.0])
    np.testing.assert_allclose(state.position, [34.0, 10.0], atol=1e-6)
    assert not bool(info["collided"])

    _, state, _, _, info = _step([5.0, 30.0], [-3.0, 0.0])
    np.testing.assert_allclose(state.position, [4.3, 30.0], atol=1e-6)
    assert bool(info["collided"])


def test_door_lip_collision_fixture():
    _, state, _, _, info = _step([32.0, 10.0], [0.0, 5.0])
    np.testing.assert_allclose(state.position, [32.0, 14.0], atol=1e-6)
    assert bool(info["collided"])


def test_reset_starts_across_rooms_and_random_layout_is_valid():
    env = PLDMTwoRoomsEnv()
    params = PLDMParams(randomize_layout=True)
    _, state = jax.jit(env.reset)(jax.random.key(3), params)
    assert (float(state.position[0]) < float(state.wall_x)) != (
        float(state.target_position[0]) < float(state.wall_x)
    )
    assert 20 <= float(state.wall_x) < 45
    assert 10 <= float(state.door_y) < 55


def test_pldm_jit_and_vmap_with_stochastic_collision():
    env = PLDMTwoRoomsEnv()
    params = env.default_params
    batch_size = 128
    keys = jax.random.split(jax.random.key(5), batch_size)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    observations, states = reset(keys, params)
    actions = jnp.tile(jnp.asarray([1.0, 0.0]), (batch_size, 1))
    observations, states, rewards, dones, info = step(keys, states, actions, params)
    assert observations.shape == (batch_size, 2)
    assert states.position.shape == (batch_size, 2)
    assert info["collided"].shape == (batch_size,)
    assert bool(jnp.all(jnp.isfinite(states.position)))
