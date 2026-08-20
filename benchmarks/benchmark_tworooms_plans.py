"""Benchmark final-only open-loop plan evaluation for Two-Room."""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

import envx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-environments", type=int, default=1024)
    parser.add_argument(
        "--num-plans",
        type=int,
        default=1,
        help="Plans per initial state (1 gives 1024 total rollouts by default).",
    )
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.num_environments, args.num_plans, args.horizon, args.repeats) < 1:
        parser.error("all size and repeat arguments must be positive")

    env, params = envx.make(
        "two-rooms",
        num_envs=args.num_environments,
        observation_type="state",
    )
    reset_key, action_key = jax.random.split(jax.random.key(args.seed))
    _, initial_states = env.reset(reset_key, params)
    action_plans = jax.random.uniform(
        action_key,
        (args.num_environments, args.num_plans, args.horizon, 2),
        minval=-1.0,
        maxval=1.0,
    )
    jax.block_until_ready((initial_states, action_plans))

    start = time.perf_counter()
    result = env.rollout_plans(initial_states, action_plans, params)
    jax.block_until_ready(result)
    compile_and_first_run = time.perf_counter() - start

    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = env.rollout_plans(initial_states, action_plans, params)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)

    num_rollouts = args.num_environments * args.num_plans
    num_transitions = num_rollouts * args.horizon
    median = statistics.median(timings)
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print(f"initial environments: {args.num_environments:,}")
    print(f"plans per environment: {args.num_plans:,}")
    print(f"horizon: {args.horizon:,}")
    print(f"parallel rollouts: {num_rollouts:,}")
    print(f"transitions per call: {num_transitions:,}")
    print(f"compile + first call: {compile_and_first_run:.6f} s")
    print("steady calls: " + ", ".join(f"{timing:.6f} s" for timing in timings))
    print(f"steady median: {median:.6f} s")
    print(f"rollouts/s: {num_rollouts / median:,.0f}")
    print(f"steps/s: {num_transitions / median:,.0f}")
    print(f"last observation: {result.last_observation.shape}")
    print(f"result success: {result.success.shape}")
    print(f"result distance: {result.info['distance_to_target'].shape}")
    print(f"final successes: {int(jnp.sum(result.success)):,}")
    print(f"mean final distance: {float(jnp.mean(result.info['distance_to_target'])):.6f}")


if __name__ == "__main__":
    main()
