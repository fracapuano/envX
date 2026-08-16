import jax
import jax.numpy as jnp
import pytest

from envx.pusht import PushTEnv, make


def test_make_matches_gymnax_constructor_pattern():
    env, params = make()
    assert isinstance(env, PushTEnv)
    assert params == env.default_params


def test_reset_and_spaces_match_canonical_state_contract():
    env = PushTEnv()
    params = env.default_params
    observation, state = env.reset(jax.random.key(0), params)

    assert observation.shape == (5,)
    assert observation.dtype == jnp.float32
    assert bool(env.action_space(params).contains(jnp.asarray((0.0, 512.0))))
    assert bool(env.observation_space(params).contains(observation))
    assert state.time.dtype == jnp.int32


def test_dataset_state_reset_is_exact():
    env = PushTEnv()
    row = jnp.asarray((12.0, 34.0, 210.0, 220.0, -0.25))
    observation, state = env.reset_from_state(row)

    assert jnp.array_equal(state.agent_pos, row[:2])
    assert jnp.array_equal(state.block_pos, row[2:4])
    assert state.block_angle == row[4]
    assert jnp.allclose(observation[:4], row[:4])
    assert observation[4] == pytest.approx(float(2.0 * jnp.pi - 0.25))


def test_legacy_random_reset_preserves_pre_rotation_center_of_mass():
    env = PushTEnv()
    params = env.default_params
    _, legacy = env.reset(jax.random.key(42), params)
    _, direct = env.reset(jax.random.key(42), params.replace(legacy_reset=False))

    expected_shift = jnp.asarray((0.0, 45.0)) - jnp.asarray(
        (
            -45.0 * jnp.sin(direct.block_angle),
            45.0 * jnp.cos(direct.block_angle),
        )
    )
    assert jnp.allclose(legacy.block_pos, direct.block_pos + expected_shift, atol=1e-5)


@pytest.mark.parametrize(
    ("observation_type", "expected_shape"),
    [
        ("state", (5,)),
        ("keypoints", (20,)),
        ("pixels", (32, 32, 3)),
    ],
)
def test_array_observation_types_jit(observation_type, expected_shape):
    env = PushTEnv(observation_type=observation_type, observation_size=32)
    observation, state = env.reset(jax.random.key(0))
    rendered = jax.jit(env.get_obs)(state, env.default_params)
    assert observation.shape == expected_shape
    assert rendered.shape == expected_shape


@pytest.mark.parametrize("observation_type", ["environment_state_agent_pos", "pixels_agent_pos"])
def test_dictionary_observation_types_work_with_auto_reset(observation_type):
    env = PushTEnv(observation_type=observation_type, observation_size=24)
    params = env.default_params.replace(max_steps_in_episode=1)
    _, state = env.reset(jax.random.key(0), params)
    observation, _state, _reward, done, info = env.step(
        jax.random.key(1), state, state.agent_pos, params
    )
    assert isinstance(observation, dict)
    assert bool(done)
    assert bool(info["truncated"])


def test_success_reward_and_gymnax_auto_reset():
    env = PushTEnv()
    params = env.default_params
    row = jnp.asarray((40.0, 40.0, params.goal_x, params.goal_y, params.goal_angle))
    _, state = env.reset_from_state(row, params)

    raw_observation, raw_state, reward, done, info = env.step_env(
        jax.random.key(0), state, state.agent_pos, params
    )
    assert float(reward) == pytest.approx(1.0)
    assert bool(done)
    assert bool(info["is_success"])
    assert jnp.allclose(raw_observation[2:4], row[2:4])
    assert raw_state.time == 1

    _observation, reset_state, reward, done, info = env.step(
        jax.random.key(0), state, state.agent_pos, params
    )
    assert float(reward) == pytest.approx(1.0)
    assert bool(done)
    assert reset_state.time == 0


def test_vmap_jit_runs_independent_batches():
    env = PushTEnv()
    params = env.default_params
    batch_size = 64
    keys = jax.random.split(jax.random.key(0), batch_size)
    reset_batch = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    observations, states = reset_batch(keys, params)
    actions = jnp.full((batch_size, 2), 256.0, dtype=jnp.float32)
    step_batch = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    next_observations, next_states, rewards, dones, _info = step_batch(
        keys, states, actions, params
    )

    assert observations.shape == (batch_size, 5)
    assert next_observations.shape == (batch_size, 5)
    assert next_states.agent_pos.shape == (batch_size, 2)
    assert rewards.shape == dones.shape == (batch_size,)
