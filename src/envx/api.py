"""Common batch-first, Gymnax-shaped interface used by envX."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from gymnax.environments import spaces

Array = jax.Array


class EmptyParams(NamedTuple):
    """Empty dynamic-parameter pytree for environments configured statically."""


class Rollout(NamedTuple):
    """Common time-major trajectory returned by every envX environment."""

    observation: Any
    action: Array
    reward: Array
    done: Array
    info: dict[str, Array]


def _standardize_info(info: dict[str, Array], done: Array) -> dict[str, Array]:
    """Add the four common termination fields without changing native fields."""

    standardized = dict(info)
    terminated = standardized.get("terminated", done)
    truncated = standardized.get("truncated", jnp.zeros_like(done))
    success = standardized.get("success", standardized.get("is_success", terminated))
    discount = standardized.get("discount", jnp.where(done, 0.0, 1.0).astype(jnp.float32))
    standardized.update(
        {
            "discount": discount,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
        }
    )
    return standardized


class VmapEnv:
    """Turn a scalar pure-JAX Gymnax environment into one compiled batch.

    The wrapper intentionally calls ``step_env`` instead of Gymnax's
    auto-resetting ``step``. This makes terminal states explicit and gives all
    four envX families the same transition semantics.
    """

    backend = "jax"

    def __init__(self, env: Any, num_envs: int):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.unwrapped = env
        self.num_envs = int(num_envs)
        self.name = env.name
        self.default_params = env.default_params
        self._action_shape = tuple(env.action_space(self.default_params).shape)

        reset_batch = jax.vmap(env.reset_env, in_axes=(0, None))
        step_batch = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))

        def reset_impl(key: Array, params: Any):
            return reset_batch(jax.random.split(key, self.num_envs), params)

        def step_impl(key: Array, state: Any, action: Array, params: Any):
            keys = jax.random.split(key, self.num_envs)
            observation, next_state, reward, done, info = step_batch(keys, state, action, params)
            info = _standardize_info(info, done)
            done = info["terminated"] | info["truncated"]
            return observation, next_state, reward, done, info

        def rollout_impl(key: Array, state: Any, actions: Array, params: Any):
            keys = jax.random.split(key, actions.shape[0])

            def scan_step(carry: Any, inputs: tuple[Array, Array]):
                step_key, action = inputs
                output = step_impl(step_key, carry, action, params)
                observation, next_state, reward, done, info = output
                transition = Rollout(observation, action, reward, done, info)
                return next_state, transition

            return jax.lax.scan(scan_step, state, (keys, actions))

        self._compiled_reset = jax.jit(reset_impl)
        self._compiled_step = jax.jit(step_impl)
        self._compiled_rollout = jax.jit(rollout_impl)

    def reset(self, key: Array, params: Any | None = None):
        """Reset every world from one root PRNG key."""

        return self._compiled_reset(key, params or self.default_params)

    def step(
        self,
        key: Array,
        state: Any,
        action: Array,
        params: Any | None = None,
    ):
        """Step every world without silently auto-resetting completed worlds."""

        expected = (self.num_envs, *self._action_shape)
        if action.shape != expected:
            raise ValueError(f"action must have shape {expected}, got {action.shape}.")
        return self._compiled_step(key, state, action, params or self.default_params)

    def rollout(
        self,
        key: Array,
        state: Any,
        actions: Array,
        params: Any | None = None,
    ) -> tuple[Any, Rollout]:
        """Run a time-major action tensor inside one compiled ``lax.scan``."""

        expected = (self.num_envs, *self._action_shape)
        if actions.ndim != len(expected) + 1 or actions.shape[1:] != expected:
            raise ValueError(
                f"actions must have shape (time, {', '.join(map(str, expected))}), "
                f"got {actions.shape}."
            )
        if actions.shape[0] < 1:
            raise ValueError("a rollout must contain at least one action.")
        return self._compiled_rollout(key, state, actions, params or self.default_params)

    def sample_actions(self, key: Array, params: Any | None = None) -> Array:
        """Sample one action independently for every world."""

        space = self.action_space(params)
        return jax.random.uniform(
            key,
            (self.num_envs, *self._action_shape),
            minval=jnp.asarray(space.low),
            maxval=jnp.asarray(space.high),
        ).astype(space.dtype)

    def action_space(self, params: Any | None = None):
        return self.unwrapped.action_space(params or self.default_params)

    def observation_space(self, params: Any | None = None):
        return self.unwrapped.observation_space(params or self.default_params)

    def render(self, state: Any, params: Any | None = None):
        """Batch-render pure-JAX environments with the native rasterizer."""

        render_one = lambda current: self.unwrapped.render(  # noqa: E731
            current, params or self.default_params
        )
        return jax.vmap(render_one)(state)


class ReacherAdapter:
    """Expose the fixed-batch Reacher implementation through the envX API."""

    backend = "mjx"

    def __init__(self, env: Any):
        self.unwrapped = env
        self.name = "reacher"
        self.num_envs = env.num_worlds
        self.default_params = EmptyParams()

    def reset(self, key: Array, params: EmptyParams | None = None):
        del params
        state = self.unwrapped.reset(jax.random.split(key, self.num_envs))
        return state.obs, state

    def step(
        self,
        key: Array,
        state: Any,
        action: Array,
        params: EmptyParams | None = None,
    ):
        del key, params
        state = self.unwrapped.step(state, action)
        info = {
            "discount": state.discount,
            "distance": state.distance,
            "success": state.success,
            "terminated": state.terminated,
            "truncated": state.truncated,
        }
        return state.obs, state, state.reward, state.done, info

    def rollout(
        self,
        key: Array,
        state: Any,
        actions: Array,
        params: EmptyParams | None = None,
    ) -> tuple[Any, Rollout]:
        del key, params
        final_state, native = self.unwrapped.rollout(state, actions)
        done = native.terminated | native.truncated
        info = {
            "discount": native.discount,
            "distance": native.distance,
            "success": native.success,
            "terminated": native.terminated,
            "truncated": native.truncated,
        }
        return final_state, Rollout(native.obs, native.action, native.reward, done, info)

    def sample_actions(self, key: Array, params: EmptyParams | None = None) -> Array:
        del params
        return self.unwrapped.sample_actions(key)

    def action_space(self, params: EmptyParams | None = None):
        del params
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: EmptyParams | None = None):
        del params
        if self.unwrapped.observation_type == "states":
            return spaces.Box(-jnp.inf, jnp.inf, shape=(6,), dtype=jnp.float32)
        return spaces.Box(
            0,
            255,
            shape=(*self.unwrapped.image_size, 3),
            dtype=jnp.uint8,
        )


class CubeAdapter:
    """Expose OGBench Cube MJX through the common envX API."""

    backend = "mjx-warp"

    def __init__(self, env: Any, task_ids: Any = 0):
        self.unwrapped = env
        self.name = "cube"
        self.num_envs = env.num_envs
        self.default_params = env.default_params
        self.task_ids = task_ids

    def reset(self, key: Array, params: Any | None = None):
        return self.unwrapped.reset(key, params or self.default_params, task_ids=self.task_ids)

    def step(
        self,
        key: Array,
        state: Any,
        action: Array,
        params: Any | None = None,
    ):
        output = self.unwrapped.step(key, state, action, params or self.default_params)
        observation, next_state, reward, done, info = output
        info = _standardize_info(info, done)
        done = info["terminated"] | info["truncated"]
        return observation, next_state, reward, done, info

    def rollout(
        self,
        key: Array,
        state: Any,
        actions: Array,
        params: Any | None = None,
    ) -> tuple[Any, Rollout]:
        final_state, native = self.unwrapped.rollout(
            key, state, actions, params or self.default_params
        )
        info = _standardize_info(native.info, native.done)
        done = info["terminated"] | info["truncated"]
        return final_state, Rollout(native.observation, actions, native.reward, done, info)

    def sample_actions(self, key: Array, params: Any | None = None) -> Array:
        del params
        return self.unwrapped.sample_actions(key)

    def action_space(self, params: Any | None = None):
        del params
        return spaces.Box(-1.0, 1.0, shape=(5,), dtype=jnp.float32)

    def observation_space(self, params: Any | None = None):
        del params
        if self.unwrapped.observation_type == "states":
            return spaces.Box(
                -jnp.inf,
                jnp.inf,
                shape=self.unwrapped.observation_shape,
                dtype=jnp.float32,
            )
        return spaces.Box(
            0,
            255,
            shape=self.unwrapped.observation_shape,
            dtype=jnp.uint8,
        )


__all__ = (
    "CubeAdapter",
    "EmptyParams",
    "ReacherAdapter",
    "Rollout",
    "VmapEnv",
)
