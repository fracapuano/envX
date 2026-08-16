"""Small, fixed-shape 2-D rigid-body solver used by PushT-JAX.

Only the contacts required by PushT are implemented: one kinematic circle
against one dynamic concave T, and the dynamic T against four static walls.
This deliberately narrow solver is easier for XLA to optimize than a general
physics engine and contains no host callbacks.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from envx.pusht.geometry import (
    _TEE_VERTICES,
    inverse_transform_points,
    origin_from_center_of_mass,
    point_in_tee,
    rotation_matrix,
    tee_center_of_mass,
    tee_vertices,
    transform_points,
)

Array = jax.Array


def cross_2d(first: Array, second: Array) -> Array:
    """Scalar two-dimensional cross product."""
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def perpendicular(vector: Array) -> Array:
    """Return the vector rotated counter-clockwise by 90 degrees."""
    return jnp.stack((-vector[..., 1], vector[..., 0]), axis=-1)


def _apply_contact(
    state: Any,
    point: Array,
    normal: Array,
    penetration: Array,
    other_velocity: Array,
    params: Any,
) -> Any:
    """Resolve one frictionless contact against the dynamic T.

    ``normal`` points from the T toward the other shape. Translation and
    rotation corrections use the same effective mass as the velocity impulse.
    """
    center = tee_center_of_mass(state.block_pos, state.block_angle)
    lever = point - center
    lever_cross_normal = cross_2d(lever, normal)
    effective_mass_inverse = 1.0 + lever_cross_normal**2 / params.block_inertia

    correction_impulse = (
        params.position_correction
        * jnp.maximum(penetration - params.collision_slop, 0.0)
        / effective_mass_inverse
    )
    corrected_center = center - correction_impulse * normal
    corrected_angle = (
        state.block_angle - correction_impulse * lever_cross_normal / params.block_inertia
    )
    corrected_origin = origin_from_center_of_mass(corrected_center, corrected_angle)

    contact_velocity = state.block_vel + state.block_angular_vel * perpendicular(lever)
    relative_velocity = other_velocity - contact_velocity
    normal_velocity = jnp.dot(relative_velocity, normal)
    velocity_impulse = jnp.maximum(-normal_velocity / effective_mass_inverse, 0.0)
    impulse = velocity_impulse * normal

    return state.replace(
        block_pos=corrected_origin,
        block_angle=corrected_angle,
        block_vel=state.block_vel - impulse,
        block_angular_vel=(
            state.block_angular_vel - cross_2d(lever, impulse) / params.block_inertia
        ),
        n_contacts=state.n_contacts + jnp.asarray(1, dtype=jnp.int32),
    )


def _resolve_agent_contact(state: Any, params: Any) -> Any:
    """Resolve the circle/T collision using the T's concave exterior boundary."""
    local_agent = inverse_transform_points(state.agent_pos, state.block_pos, state.block_angle)
    vertices = jnp.asarray(_TEE_VERTICES, dtype=jnp.float32)
    edge_ends = jnp.roll(vertices, -1, axis=0)
    edges = edge_ends - vertices
    edge_length_squared = jnp.sum(edges**2, axis=1)
    edge_parameter = jnp.sum((local_agent - vertices) * edges, axis=1) / edge_length_squared
    edge_parameter = jnp.clip(edge_parameter, 0.0, 1.0)
    closest_points = vertices + edge_parameter[:, None] * edges
    deltas = local_agent - closest_points
    squared_distances = jnp.sum(deltas**2, axis=1)
    edge_index = jnp.argmin(squared_distances)
    closest = closest_points[edge_index]
    delta = deltas[edge_index]
    distance = jnp.sqrt(jnp.maximum(squared_distances[edge_index], 1e-12))

    selected_edge = edges[edge_index]
    outward = jnp.array((selected_edge[1], -selected_edge[0]))
    outward = outward / jnp.linalg.norm(outward)
    inside = point_in_tee(local_agent)
    outside_normal = jnp.where(distance > 1e-6, delta / distance, outward)
    local_normal = jnp.where(inside, outward, outside_normal)
    penetration = jnp.where(inside, params.agent_radius + distance, params.agent_radius - distance)

    world_normal = local_normal @ rotation_matrix(state.block_angle).T
    world_point = transform_points(closest, state.block_pos, state.block_angle)
    colliding = penetration > 0.0
    return jax.lax.cond(
        colliding,
        lambda current: _apply_contact(
            current,
            world_point,
            world_normal,
            penetration,
            current.agent_vel,
            params,
        ),
        lambda current: current,
        state,
    )


def _wall_contact_data(vertices: Array, wall_index: Array, params: Any) -> tuple[Array, ...]:
    """Return contact point, normal, and penetration for one wall."""
    minimum_x_index = jnp.argmin(vertices[:, 0])
    maximum_x_index = jnp.argmax(vertices[:, 0])
    minimum_y_index = jnp.argmin(vertices[:, 1])
    maximum_y_index = jnp.argmax(vertices[:, 1])

    points = jnp.stack(
        (
            vertices[minimum_x_index],
            vertices[maximum_x_index],
            vertices[minimum_y_index],
            vertices[maximum_y_index],
        )
    )
    normals = jnp.asarray(
        ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)),
        dtype=jnp.float32,
    )
    penetrations = jnp.stack(
        (
            params.wall_min - vertices[minimum_x_index, 0],
            vertices[maximum_x_index, 0] - params.wall_max,
            params.wall_min - vertices[minimum_y_index, 1],
            vertices[maximum_y_index, 1] - params.wall_max,
        )
    )
    return points[wall_index], normals[wall_index], penetrations[wall_index]


def _resolve_wall_contact(wall_index: Array, state: Any, params: Any) -> Any:
    vertices = tee_vertices(state.block_pos, state.block_angle)
    point, normal, penetration = _wall_contact_data(vertices, wall_index, params)
    return jax.lax.cond(
        penetration > 0.0,
        lambda current: _apply_contact(
            current,
            point,
            normal,
            penetration,
            jnp.zeros(2, dtype=jnp.float32),
            params,
        ),
        lambda current: current,
        state,
    )


def substep(state: Any, action: Array, params: Any) -> Any:
    """Advance the controller and physics by one 100 Hz substep."""
    dt = params.dt
    acceleration = params.k_p * (action - state.agent_pos) - params.k_v * state.agent_vel
    agent_velocity = state.agent_vel + acceleration * dt
    agent_position = state.agent_pos + agent_velocity * dt

    # Chipmunk integrates positions before damping velocities. Its damping is a
    # per-second retention factor, hence the exponent by dt.
    center = tee_center_of_mass(state.block_pos, state.block_angle)
    center = center + state.block_vel * dt
    block_angle = state.block_angle + state.block_angular_vel * dt
    block_origin = origin_from_center_of_mass(center, block_angle)
    damping = jnp.where(
        params.damping <= 0.0,
        0.0,
        jnp.power(params.damping, dt),
    )

    state = state.replace(
        agent_pos=agent_position,
        agent_vel=agent_velocity,
        block_pos=block_origin,
        block_angle=block_angle,
        block_vel=state.block_vel * damping,
        block_angular_vel=state.block_angular_vel * damping,
    )
    state = _resolve_agent_contact(state, params)
    return jax.lax.fori_loop(
        0,
        4,
        lambda wall_index, current: _resolve_wall_contact(wall_index, current, params),
        state,
    )


def simulate_control_step(state: Any, action: Array, params: Any) -> Any:
    """Advance one 10 Hz PushT control step."""
    state = state.replace(n_contacts=jnp.asarray(0, dtype=jnp.int32))
    state = jax.lax.fori_loop(
        0,
        params.n_substeps,
        lambda _index, current: substep(current, action, params),
        state,
    )
    contacts_per_substep = jnp.ceil(state.n_contacts / params.n_substeps).astype(jnp.int32)
    return state.replace(n_contacts=contacts_per_substep)
