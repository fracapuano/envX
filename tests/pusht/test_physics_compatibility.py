import jax
import jax.numpy as jnp

from envx.pusht import PushTEnv


def _rollout(initial_state, action, steps):
    env = PushTEnv()
    params = env.default_params
    _, state = env.reset_from_state(jnp.asarray(initial_state), params)
    rows = []
    for index in range(steps):
        observation, state, *_ = env.step_env(
            jax.random.key(index), state, jnp.asarray(action), params
        )
        rows.append(observation)
    return jnp.stack(rows)


def test_free_pd_motion_matches_reference_controller():
    rows = _rollout((256.0, 420.0, 256.0, 240.0, 0.0), (256.0, 300.0), 1)
    # Fixture collected from gym-pusht 0.1.6 / Pymunk 6.11.1.
    assert jnp.allclose(rows[0, :2], jnp.asarray((256.0, 384.51938569)), atol=1e-3)
    assert jnp.array_equal(rows[0, 2:], jnp.asarray((256.0, 240.0, 0.0)))


def test_centered_push_tracks_reference_trajectory():
    rows = _rollout((256.0, 420.0, 256.0, 240.0, 0.0), (256.0, 300.0), 20)
    indices = jnp.asarray((0, 1, 2, 4, 9, 19))
    reference = jnp.asarray(
        (
            (256.0, 384.51938569, 256.0, 240.0, 0.0),
            (256.0, 346.09068702, 256.0, 212.44915984, 0.0),
            (256.0, 323.08576985, 256.0, 189.59242165, 0.0),
            (256.0, 305.30818426, 256.0, 170.60369641, 0.0),
            (256.0, 300.12167011, 256.0, 165.13088402, 0.0),
            (256.0, 300.00006224, 256.0, 165.00006627, 0.0),
        )
    )
    assert jnp.allclose(rows[indices], reference, atol=0.25)


def test_off_center_push_tracks_reference_translation_and_rotation():
    rows = _rollout((100.0, 215.0, 200.0, 200.0, 0.0), (260.0, 215.0), 10)
    indices = jnp.asarray((0, 1, 2, 4, 9))
    reference = jnp.asarray(
        (
            (147.307486, 215.0, 220.822914, 201.105426, 0.132813034),
            (198.545751, 215.0, 278.574961, 216.745451, 0.403629386),
            (229.218974, 215.0, 305.984602, 230.437326, 0.539965601),
            (252.922421, 215.0, 325.457789, 243.669728, 0.662919731),
            (259.837773, 215.0, 330.378174, 247.653892, 0.699824951),
        )
    )
    position_error = jnp.abs(rows[indices, :4] - reference[:, :4])
    angle_error = jnp.abs(rows[indices, 4] - reference[:, 4])
    assert jnp.max(position_error) < 0.30
    assert jnp.max(angle_error) < 8e-4
