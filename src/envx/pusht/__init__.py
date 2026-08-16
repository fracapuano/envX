"""Pure-JAX PushT environment."""

from envx.pusht.environment import EnvParams, EnvState, PushTEnv
from envx.pusht.geometry import coverage, diffusion_policy_keypoints, tee_vertices
from envx.pusht.trajectory import dataset_state, reset_from_dataset_state, rollout


def make(**kwargs):
    """Construct ``(environment, default_params)`` like ``gymnax.make``."""
    env = PushTEnv(**kwargs)
    return env, env.default_params


PushT = PushTEnv

__all__ = [
    "EnvParams",
    "EnvState",
    "PushT",
    "PushTEnv",
    "coverage",
    "dataset_state",
    "diffusion_policy_keypoints",
    "make",
    "reset_from_dataset_state",
    "rollout",
    "tee_vertices",
]

__version__ = "0.1.0"
