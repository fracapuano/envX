# envX

`envX` provides one batch-first JAX interface for PushT, Two-Room, Reacher,
and OGBench Cube. Use `observation_type="state"` for physics-only workloads;
rendering is disabled automatically.

## Fast parallel plan rollouts

Every environment exposes the same final-only API:

```python
import jax
import jax.numpy as jnp

import envx

num_initial_states = 1024
num_plans = 1
horizon = 25

env, params = envx.make(
    "two-rooms",  # "pusht", "reacher", or "cube" work the same way
    num_envs=num_initial_states,
    observation_type="state",
)

reset_key, action_key = jax.random.split(jax.random.key(0))
_, initial_states = env.reset(reset_key, params)
action_space = env.action_space(params)
action_plans = jax.random.uniform(
    action_key,
    (
        num_initial_states,
        num_plans,
        horizon,
        *action_space.shape,
    ),
    minval=jnp.asarray(action_space.low),
    maxval=jnp.asarray(action_space.high),
)

# The first call compiles. Later calls with the same shapes reuse the executable.
result = env.rollout_plans(initial_states, action_plans, params)
jax.block_until_ready(result)

print(result.last_observation.shape)  # (1024, 1, observation_size)
print(result.reward.shape)  # (1024, 1)
print(result.done.shape)  # (1024, 1)
print(result.success.shape)  # (1024, 1)
print(result.info)
```

`action_plans` has shape `(initial_state, plan, time, *action_shape)`. All
initial-state/plan pairs are evaluated in parallel, while the time axis runs
inside one compiled `jax.lax.scan`. Only the final observation and score are
returned, so intermediate trajectories are not allocated.

The common result fields are `last_observation`, `reward`, `done`, `success`,
and `info`. Task-specific final metrics remain in `info`, including
`coverage` for PushT, `distance_to_target` for Two-Room and Reacher, and
`cube_success` for Cube.
