"""Environment registry for the unified envX entry point."""

from __future__ import annotations

from typing import Any

from envx.api import CubeAdapter, ReacherAdapter, VmapEnv

_ALIASES = {
    "cube": "cube",
    "cube-v0": "cube",
    "ogbench-cube": "cube",
    "pusht": "pusht",
    "pusht-v0": "pusht",
    "reacher": "reacher",
    "reacher-v0": "reacher",
    "two-rooms": "two-rooms",
    "two-rooms-v1": "two-rooms",
    "tworooms": "two-rooms",
    "two-rooms-pldm": "two-rooms-pldm",
    "tworooms-pldm": "two-rooms-pldm",
}

REGISTERED_ENVIRONMENTS = ("pusht", "two-rooms", "reacher", "cube")
ENVIRONMENT_VARIANTS = (*REGISTERED_ENVIRONMENTS, "two-rooms-pldm")


def canonical_name(name: str) -> str:
    """Resolve a case-insensitive public name to its canonical registry key."""

    normalized = name.strip().lower().replace("_", "-")
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        available = ", ".join(ENVIRONMENT_VARIANTS)
        raise ValueError(f"Unknown environment {name!r}; choose one of: {available}.") from exc


def make(name: str, *, num_envs: int = 1, **kwargs: Any):
    """Construct ``(env, params)`` using the common batch-first API.

    Args:
        name: One of ``pusht``, ``two-rooms``, ``reacher``, or ``cube``.
        num_envs: Static number of parallel worlds compiled together.
        **kwargs: Environment-specific static configuration.
    """

    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    name = canonical_name(name)

    if name == "pusht":
        from envx.pusht import PushTEnv

        env = VmapEnv(PushTEnv(**kwargs), num_envs)
    elif name == "two-rooms":
        from envx.tworooms import TwoRoomsEnv

        env = VmapEnv(TwoRoomsEnv(**kwargs), num_envs)
    elif name == "two-rooms-pldm":
        from envx.tworooms import PLDMTwoRoomsEnv

        env = VmapEnv(PLDMTwoRoomsEnv(**kwargs), num_envs)
    elif name == "reacher":
        from envx.reacher import ReacherEnv

        observation_type = kwargs.setdefault("observation_type", "states")
        kwargs.setdefault("render", observation_type == "pixels")
        env = ReacherAdapter(ReacherEnv(num_worlds=num_envs, **kwargs))
    else:
        from envx.cube import OGBenchCubeJaxEnv

        task_ids = kwargs.pop("task_ids", 0)
        env = CubeAdapter(OGBenchCubeJaxEnv(num_envs=num_envs, **kwargs), task_ids=task_ids)
    return env, env.default_params


def registered_environments(*, include_variants: bool = False) -> tuple[str, ...]:
    """Return stable canonical names accepted by :func:`make`."""

    return ENVIRONMENT_VARIANTS if include_variants else REGISTERED_ENVIRONMENTS


__all__ = (
    "ENVIRONMENT_VARIANTS",
    "REGISTERED_ENVIRONMENTS",
    "canonical_name",
    "make",
    "registered_environments",
)
