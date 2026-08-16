from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import envx


def _assert_common_step(env, params, observation, state):
    action = env.sample_actions(jax.random.key(2), params)
    output = env.step(jax.random.key(3), state, action, params)
    next_observation, next_state, reward, done, info = output

    assert jax.tree.leaves(observation)[0].shape[0] == env.num_envs
    assert jax.tree.leaves(next_observation)[0].shape[0] == env.num_envs
    assert reward.shape == (env.num_envs,)
    assert done.shape == (env.num_envs,)
    assert action.shape == (env.num_envs, *env.action_space(params).shape)
    assert next_state is not None
    assert {"discount", "success", "terminated", "truncated"} <= info.keys()
    np.testing.assert_array_equal(done, info["terminated"] | info["truncated"])
    return next_state


@pytest.mark.parametrize(
    ("name", "kwargs", "observation_shape", "action_shape"),
    [
        ("pusht", {}, (3, 5), (2,)),
        ("two-rooms", {}, (3, 10), (2,)),
        ("two-rooms-pldm", {}, (3, 2), (2,)),
    ],
)
def test_pure_jax_registry_environments_share_batch_api(
    name, kwargs, observation_shape, action_shape
):
    env, params = envx.make(name, num_envs=3, **kwargs)
    observation, state = env.reset(jax.random.key(1), params)

    assert observation.shape == observation_shape
    assert env.action_space(params).shape == action_shape
    state = _assert_common_step(env, params, observation, state)

    action_keys = jax.random.split(jax.random.key(4), 2)
    actions = jax.vmap(lambda key: env.sample_actions(key, params))(action_keys)
    final_state, trajectory = env.rollout(jax.random.key(5), state, actions, params)
    jax.block_until_ready((final_state, trajectory))
    assert trajectory.observation.shape[:2] == (2, 3)
    assert trajectory.action.shape == (2, 3, *action_shape)
    assert trajectory.reward.shape == (2, 3)
    assert trajectory.done.shape == (2, 3)


def test_common_pure_jax_step_preserves_terminal_state():
    env, params = envx.make("pusht", num_envs=2)
    params = params.replace(max_steps_in_episode=1)
    _, state = env.reset(jax.random.key(10), params)
    action = env.sample_actions(jax.random.key(11), params)
    _, next_state, _, done, info = env.step(jax.random.key(12), state, action, params)

    np.testing.assert_array_equal(done, True)
    np.testing.assert_array_equal(info["truncated"], True)
    np.testing.assert_array_equal(next_state.time, 1)


def test_reacher_registry_adapter_uses_the_same_contract():
    env, params = envx.make(
        "reacher",
        num_envs=2,
        physics_backend="jax",
        observation_type="states",
        render=False,
    )
    observation, state = env.reset(jax.random.key(20), params)
    assert observation.shape == (2, 6)
    state = _assert_common_step(env, params, observation, state)

    actions = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    _, trajectory = env.rollout(jax.random.key(21), state, actions, params)
    assert trajectory.observation.shape == (2, 2, 6)
    assert trajectory.reward.shape == (2, 2)


def test_cube_registry_adapter_uses_the_same_contract():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        env, params = envx.make(
            "cube",
            num_envs=1,
            env_type="single",
            observation_type="states",
            task_ids=2,
        )
    observation, state = env.reset(jax.random.key(30), params)
    assert observation.shape == (1, 28)
    state = _assert_common_step(env, params, observation, state)

    actions = jnp.zeros((1, 1, 5), dtype=jnp.float32)
    _, trajectory = env.rollout(jax.random.key(31), state, actions, params)
    assert trajectory.observation.shape == (1, 1, 28)
    assert trajectory.reward.shape == (1, 1)


def test_registry_names_aliases_and_errors():
    assert envx.registered_environments() == (
        "pusht",
        "two-rooms",
        "reacher",
        "cube",
    )
    assert "two-rooms-pldm" in envx.registered_environments(include_variants=True)
    env, _ = envx.make("PushT-v0", num_envs=1)
    assert env.name == "PushT-JAX-v0"
    with pytest.raises(ValueError, match="Unknown environment"):
        envx.make("not-an-environment")
    with pytest.raises(ValueError, match="num_envs must be positive"):
        envx.make("pusht", num_envs=0)


def test_cube_assets_are_packaged_beside_the_model_builder():
    description_dir = (
        Path(__file__).parents[1] / "src" / "envx" / "cube" / "_vendor" / "descriptions"
    )
    required = (
        description_dir / "floor_wall.xml",
        description_dir / "cube_inner.xml",
        description_dir / "universal_robots_ur5e" / "ur5e.xml",
        description_dir / "robotiq_2f85" / "2f85.xml",
        description_dir / "universal_robots_ur5e" / "assets" / "upperarm_3.obj",
    )
    assert all(path.is_file() for path in required)
