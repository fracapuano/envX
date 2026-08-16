"""Pure-JAX Two Rooms environments."""

from envx.tworooms.environment import (
    EnvParams,
    EnvState,
    TwoRooms,
    TwoRoomsEnv,
    state_from_proprio,
)
from envx.tworooms.pldm import (
    PLDMParams,
    PLDMState,
    PLDMTwoRoomsEnv,
    pldm_state_from_trajectory,
    resolve_pldm_collision,
)
from envx.tworooms.policies import pldm_random_action, random_action, weak_expert_action
from envx.tworooms.trajectory import (
    Trajectory,
    normalize_pldm_location,
    normalize_pldm_pixels,
    render_pldm_locations,
    rollout,
    state_from_swm_observation,
    swm_observation,
    unnormalize_pldm_location,
)


def make(**kwargs):
    """Construct current ``(environment, default_params)`` like ``gymnax.make``."""
    env = TwoRoomsEnv(**kwargs)
    return env, env.default_params


def make_pldm(**kwargs):
    """Construct classic PLDM ``(environment, default_params)``."""
    env = PLDMTwoRoomsEnv(**kwargs)
    return env, env.default_params


__all__ = [
    "EnvParams",
    "EnvState",
    "PLDMParams",
    "PLDMState",
    "PLDMTwoRoomsEnv",
    "Trajectory",
    "TwoRooms",
    "TwoRoomsEnv",
    "make",
    "make_pldm",
    "normalize_pldm_location",
    "normalize_pldm_pixels",
    "pldm_random_action",
    "pldm_state_from_trajectory",
    "random_action",
    "render_pldm_locations",
    "resolve_pldm_collision",
    "rollout",
    "state_from_proprio",
    "state_from_swm_observation",
    "swm_observation",
    "unnormalize_pldm_location",
    "weak_expert_action",
]

__version__ = "0.1.0"
