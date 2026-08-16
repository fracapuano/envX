"""Run one compiled random-policy trajectory through the common envX API."""

from __future__ import annotations

import argparse

import jax

import envx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=envx.registered_environments())
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env, params = envx.make(args.environment, num_envs=args.num_envs)
    reset_key, action_key, rollout_key = jax.random.split(jax.random.key(args.seed), 3)
    observation, state = env.reset(reset_key, params)
    action_keys = jax.random.split(action_key, args.steps)
    actions = jax.vmap(lambda key: env.sample_actions(key, params))(action_keys)
    final_state, trajectory = env.rollout(rollout_key, state, actions, params)
    jax.block_until_ready((final_state, trajectory))

    print(f"environment: {args.environment}")
    print(f"backend: {env.backend}")
    print(f"reset observation: {jax.tree.map(lambda x: x.shape, observation)}")
    print(f"trajectory reward: {trajectory.reward.shape}")
    print(f"completed transitions: {int(trajectory.done.sum())}")


if __name__ == "__main__":
    main()
