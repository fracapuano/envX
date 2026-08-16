import jax
import jax.numpy as jnp
import numpy as np

from envx.tworooms import EnvParams, TwoRoomsEnv, state_from_proprio, weak_expert_action


def _numpy_collision(
    position,
    proposed,
    *,
    radius=7.0,
    wall_axis=1,
    wall_thickness=10,
    door_positions=(49.0, 49.0, 49.0),
    door_sizes=(14.0, 14.0, 14.0),
    num_doors=1,
):
    bounded = np.clip(proposed, 14.0 + radius, 224.0 - 14.0 - radius).astype(np.float32)
    half = wall_thickness // 2
    low = 112.0 - half - radius
    high = 112.0 + half + radius
    if wall_axis == 1:
        axis, along = 0, 1
    else:
        axis, along = 1, 0
    in_door = any(
        door_positions[i] - door_sizes[i] - 1.75
        <= bounded[along]
        <= door_positions[i] + door_sizes[i] + 1.75
        for i in range(num_doors)
    )
    started_low = position[axis] < 112.0
    if started_low and bounded[axis] > low and not in_door:
        bounded[axis] = low - 0.5
    elif not started_low and bounded[axis] < high and not in_door:
        bounded[axis] = high + 0.5
    return bounded


def test_reset_and_spaces_are_gymnax_compatible():
    env = TwoRoomsEnv()
    params = env.default_params
    observation, state = jax.jit(env.reset)(jax.random.key(0), params)

    assert observation.shape == (10,)
    assert observation.dtype == jnp.float32
    assert env.action_space(params).shape == (2,)
    assert env.observation_space(params).shape == (10,)
    assert bool(env.observation_space(params).contains(observation))
    assert int(state.time) == 0

    half = int(state.wall_thickness) // 2
    wall_low = 112.0 - half - float(state.agent_radius)
    wall_high = 112.0 + half + float(state.agent_radius)
    axis = 0 if int(state.wall_axis) == 1 else 1
    assert (
        float(state.agent_position[axis]) < wall_low
        or float(state.agent_position[axis]) > wall_high
    )


def test_collision_matches_upstream_destination_clamp():
    env = TwoRoomsEnv()
    params = env.default_params
    state = state_from_proprio(jnp.array([95.0, 100.0]), jnp.array([190.0, 100.0]), params)

    _, state, _, _, info = env.step_env(jax.random.key(0), state, jnp.array([1.0, 0.0]), params)
    np.testing.assert_allclose(state.agent_position, [100.0, 100.0])
    assert not bool(info["collided"])

    _, state, _, _, info = env.step_env(jax.random.key(1), state, jnp.array([1.0, 0.0]), params)
    np.testing.assert_allclose(state.agent_position, [99.5, 100.0])
    assert bool(info["collided"])


def test_door_and_horizontal_layout_allow_passage():
    env = TwoRoomsEnv()
    params = env.default_params
    vertical = state_from_proprio(jnp.array([100.0, 49.0]), jnp.array([190.0, 49.0]), params)
    _, vertical, _, _, info = env.step_env(
        jax.random.key(0), vertical, jnp.array([1.0, 0.0]), params
    )
    np.testing.assert_allclose(vertical.agent_position, [105.0, 49.0])
    assert not bool(info["collided"])

    horizontal = state_from_proprio(
        jnp.array([49.0, 100.0]),
        jnp.array([49.0, 190.0]),
        params,
        wall_axis=0,
    )
    _, horizontal, _, _, info = env.step_env(
        jax.random.key(0), horizontal, jnp.array([0.0, 1.0]), params
    )
    np.testing.assert_allclose(horizontal.agent_position, [49.0, 105.0])
    assert not bool(info["collided"])


def test_border_action_clip_success_and_timeout():
    env = TwoRoomsEnv()
    params = EnvParams(max_steps_in_episode=1)
    state = state_from_proprio(jnp.array([202.0, 100.0]), jnp.array([203.0, 100.0]), params)
    _, next_state, reward, done, info = env.step_env(
        jax.random.key(0), state, jnp.array([5.0, 0.0]), params
    )
    np.testing.assert_allclose(next_state.agent_position, [203.0, 100.0])
    assert float(reward) == 0.0
    assert bool(done)
    assert bool(info["terminated"])
    assert bool(info["truncated"])


def test_formula_matches_numpy_reference_across_random_cases():
    env = TwoRoomsEnv()
    params = env.default_params
    rng = np.random.default_rng(7)
    for wall_axis in (0, 1):
        for _ in range(40):
            position = rng.uniform(21.0, 203.0, size=2).astype(np.float32)
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            state = state_from_proprio(position, [30.0, 30.0], params, wall_axis=wall_axis)
            _, next_state, _, _, _ = env.step_env(jax.random.key(0), state, action, params)
            expected = _numpy_collision(
                position,
                position + np.clip(action, -1.0, 1.0) * 5.0,
                wall_axis=wall_axis,
            )
            np.testing.assert_allclose(next_state.agent_position, expected, atol=1e-6)


def test_jit_vmap_and_randomized_layouts():
    env = TwoRoomsEnv()
    params = EnvParams(randomize_layout=True)
    batch_size = 64
    keys = jax.random.split(jax.random.key(42), batch_size)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    observations, states = reset(keys, params)
    actions = jnp.zeros((batch_size, 2), jnp.float32)
    observations, states, rewards, dones, info = step(keys, states, actions, params)

    assert observations.shape == (batch_size, 10)
    assert states.agent_position.shape == (batch_size, 2)
    assert rewards.shape == dones.shape == (batch_size,)
    assert info["pos_agent"].shape == (batch_size, 2)
    assert bool(jnp.all(states.door_sizes[:, 0] >= jnp.ceil(1.1 * states.agent_radius)))


def test_gymnax_step_auto_resets_after_success():
    env = TwoRoomsEnv()
    params = env.default_params
    state = state_from_proprio(jnp.array([50.0, 50.0]), jnp.array([50.0, 50.0]), params)
    _, selected_state, _, done, info = env.step(jax.random.key(9), state, jnp.zeros(2), params)
    assert bool(done)
    assert bool(info["is_success"])
    assert int(selected_state.time) == 0


def test_jitted_weak_expert_solves_opposite_room_fixture():
    env = TwoRoomsEnv()
    params = env.default_params
    state = state_from_proprio([60.0, 100.0], [180.0, 100.0], params)
    step = jax.jit(env.step_env)
    policy = jax.jit(weak_expert_action)
    for index in range(100):
        _, state, _, done, info = step(
            jax.random.fold_in(jax.random.key(0), index), state, policy(state), params
        )
        if bool(done):
            break
    assert bool(info["is_success"])
    assert index < 99
