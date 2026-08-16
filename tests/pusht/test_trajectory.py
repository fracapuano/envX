import jax
import jax.numpy as jnp

from envx.pusht import PushTEnv, dataset_state, reset_from_dataset_state, rollout


def test_dataset_round_trip():
    row = jnp.asarray((10.0, 20.0, 30.0, 40.0, 0.5))
    state = reset_from_dataset_state(row)
    assert jnp.array_equal(dataset_state(state), row)


def test_rollout_compiles_to_scan_and_preserves_terminal_state():
    env = PushTEnv()
    params = env.default_params
    initial = reset_from_dataset_state(jnp.asarray((256.0, 420.0, 256.0, 240.0, 0.0)))
    actions = jnp.tile(jnp.asarray((256.0, 300.0)), (20, 1))
    run = jax.jit(lambda key, state, acts: rollout(env, key, state, acts, params))
    final_state, trajectory = run(jax.random.key(0), initial, actions)

    assert trajectory.observations.shape == (20, 5)
    assert trajectory.actions.shape == (20, 2)
    assert trajectory.rewards.shape == (20,)
    assert trajectory.info["coverage"].shape == (20,)
    assert final_state.time == 20
