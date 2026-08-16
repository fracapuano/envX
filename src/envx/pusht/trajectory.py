"""Helpers for public PushT trajectory state/action arrays."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from envx.pusht.environment import EnvParams, EnvState

Array = jax.Array


@struct.dataclass
class Trajectory:
    """Stacked outputs from a JAX ``lax.scan`` rollout."""

    observations: Any
    actions: Array
    rewards: Array
    terminated: Array
    truncated: Array
    info: dict[str, Array]


def reset_from_dataset_state(state: Array) -> EnvState:
    """Create simulator state from a common PushT five-value state row."""
    state = jnp.asarray(state, dtype=jnp.float32)
    return EnvState(
        agent_pos=state[:2],
        agent_vel=jnp.zeros(2, dtype=jnp.float32),
        block_pos=state[2:4],
        block_angle=state[4],
        block_vel=jnp.zeros(2, dtype=jnp.float32),
        block_angular_vel=jnp.asarray(0.0, dtype=jnp.float32),
        last_action=state[:2],
        n_contacts=jnp.asarray(0, dtype=jnp.int32),
        time=jnp.asarray(0, dtype=jnp.int32),
    )


def dataset_state(state: EnvState) -> Array:
    """Convert simulator state to ``[agent, block, angle]`` dataset layout."""
    return jnp.concatenate(
        (state.agent_pos, state.block_pos, jnp.mod(state.block_angle, 2.0 * jnp.pi)[None])
    )


def rollout(
    env,
    key: Array,
    initial_state: EnvState,
    actions: Array,
    params: EnvParams | None = None,
    *,
    auto_reset: bool = False,
) -> tuple[EnvState, Trajectory]:
    """Roll out a target-position action sequence with one compiled scan.

    Set ``auto_reset=False`` (the default) when comparing to an offline dataset;
    it uses ``step_env`` and therefore preserves terminal states. Set it to true
    for Gymnax's standard auto-reset behavior.
    """
    if params is None:
        params = env.default_params
    keys = jax.random.split(key, actions.shape[0])

    def step(current_state: EnvState, inputs: tuple[Array, Array]):
        step_key, action = inputs
        if auto_reset:
            observation, next_state, reward, _done, info = env.step(
                step_key, current_state, action, params
            )
        else:
            observation, next_state, reward, _done, info = env.step_env(
                step_key, current_state, action, params
            )
        terminated = info["terminated"]
        truncated = info["truncated"]
        outputs = (observation, reward, terminated, truncated, info)
        return next_state, outputs

    final_state, outputs = jax.lax.scan(step, initial_state, (keys, actions))
    observations, rewards, terminated, truncated, info = outputs
    return final_state, Trajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )
