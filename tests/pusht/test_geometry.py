import jax
import jax.numpy as jnp
import pytest

from envx.pusht.geometry import coverage, diffusion_policy_keypoints, tee_vertices

GOAL_POSITION = jnp.asarray((256.0, 256.0), dtype=jnp.float32)
GOAL_ANGLE = jnp.asarray(jnp.pi / 4.0, dtype=jnp.float32)


@pytest.mark.parametrize(
    ("position", "angle", "expected"),
    [
        ((256.0, 256.0), jnp.pi / 4.0, 1.0),
        ((220.0, 270.0), -0.4, 0.38762307),
        ((300.0, 210.0), 1.2, 0.15608072),
        ((256.0, 196.0), jnp.pi / 4.0, 1.0 / 7.0),
        ((100.0, 100.0), 0.0, 0.0),
    ],
)
def test_coverage_matches_shapely_fixtures(position, angle, expected):
    actual = coverage(jnp.asarray(position), jnp.asarray(angle), GOAL_POSITION, GOAL_ANGLE)
    assert float(actual) == pytest.approx(expected, abs=2e-6)


def test_coverage_is_jittable_and_vectorized():
    positions = jnp.asarray(((256.0, 256.0), (100.0, 100.0), (220.0, 270.0)))
    angles = jnp.asarray((jnp.pi / 4.0, 0.0, -0.4))
    batched = jax.jit(jax.vmap(coverage, in_axes=(0, 0, None, None)))
    values = batched(positions, angles, GOAL_POSITION, GOAL_ANGLE)
    assert values.shape == (3,)
    assert jnp.allclose(values, jnp.asarray((1.0, 0.0, 0.38762307)), atol=2e-6)


def test_tee_vertices_are_in_public_body_origin_frame():
    vertices = tee_vertices(jnp.asarray((10.0, 20.0)), jnp.asarray(0.0))
    assert jnp.array_equal(vertices[0], jnp.asarray((-50.0, 20.0)))
    assert jnp.array_equal(vertices[4], jnp.asarray((25.0, 140.0)))


def test_diffusion_policy_keypoints_have_expected_layout():
    observation = diffusion_policy_keypoints(
        jnp.asarray((256.0, 300.0)),
        jnp.asarray(0.0),
        jnp.asarray((100.0, 200.0)),
    )
    assert observation.shape == (20,)
    assert jnp.allclose(observation[:2], jnp.asarray((265.2698, 390.0410)), atol=1e-3)
    assert jnp.array_equal(observation[-2:], jnp.asarray((100.0, 200.0)))
