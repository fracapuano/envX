"""Render a reproducible random rollout from any envX environment.

The MJX-Warp renderer uses its CPU validation path on macOS.  Training-time
pixel rendering is intended for CUDA, but this example deliberately keeps the
visual check portable.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

if platform.system() == "Darwin":
    os.environ.setdefault("WARP_DISABLE_CUDA", "1")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import envx

ENVIRONMENTS = ("pusht", "two-rooms", "reacher", "cube")


def _held_random_actions(
    key: jax.Array,
    *,
    steps: int,
    shape: tuple[int, ...],
    minimum: float,
    maximum: float,
    hold: int,
) -> jax.Array:
    """Sample a piecewise-constant random policy so motion is easy to see."""

    num_segments = (steps + hold - 1) // hold
    segments = jax.random.uniform(
        key,
        (num_segments, *shape),
        minval=minimum,
        maxval=maximum,
        dtype=jnp.float32,
    )
    return jnp.repeat(segments, hold, axis=0)[:steps]


def _collect_pixel_observations(
    name: str,
    *,
    seed: int,
    steps: int,
    image_size: int,
) -> list[np.ndarray]:
    if name == "pusht":
        env, params = envx.make(
            "pusht",
            num_envs=1,
            observation_type="pixels",
            observation_size=image_size,
        )
        actions = _held_random_actions(
            jax.random.key(seed + 1),
            steps=steps,
            shape=(1, 2),
            minimum=35.0,
            maximum=477.0,
            hold=8,
        )
    else:
        env, params = envx.make(
            "two-rooms",
            num_envs=1,
            observation_type="pixels",
            visualize_goal=False,
        )
        angles = _held_random_actions(
            jax.random.key(seed + 1),
            steps=steps,
            shape=(1, 1),
            minimum=-jnp.pi,
            maximum=jnp.pi,
            hold=10,
        )
        actions = jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1)

    observation, state = env.reset(jax.random.key(seed), params)
    frames = [np.asarray(jax.block_until_ready(observation[0]))]
    key = jax.random.key(seed + 2)
    for action in actions:
        key, step_key = jax.random.split(key)
        observation, state, _, _, _ = env.step(step_key, state, action, params)
        frames.append(np.asarray(jax.block_until_ready(observation[0])))
    return frames


def _collect_reacher(
    *,
    seed: int,
    steps: int,
    image_size: int,
) -> list[np.ndarray]:
    from mujoco import mjx

    from envx.reacher.env import _render_batch

    cpu = jax.local_devices(backend="cpu")[0]
    env, params = envx.make(
        "reacher",
        num_envs=1,
        image_size=(image_size, image_size),
        observation_type="pixels",
        physics_backend="warp",
        render=False,
        visualize_goal=False,
        devices=[cpu],
    )
    native = env.unwrapped
    _, state = env.reset(jax.random.key(seed), params)

    camera_id = native.mj_model.camera("fixed").id
    context = mjx.create_render_context(
        mjm=native.mj_model,
        nworld=native.num_worlds,
        devices=["cpu"],
        cam_res=(image_size, image_size),
        cam_active=[index == camera_id for index in range(native.mj_model.ncam)],
        use_textures=True,
        use_shadows=True,
        render_skybox=True,
        render_rgb=True,
        render_depth=False,
        enabled_geom_groups=[0, 1, 2],
    )
    context_pytree = context.pytree()

    @jax.jit
    def render(data):
        return _render_batch(native.model, data, context_pytree, camera_id)

    def capture(current_state):
        data, pixels = render(current_state.data)
        pixels = jax.block_until_ready(pixels)
        return current_state._replace(data=data), np.asarray(pixels[0])

    state, first_frame = capture(state)
    frames = [first_frame]
    actions = _held_random_actions(
        jax.random.key(seed + 1),
        steps=steps,
        shape=(1, 2),
        minimum=-1.0,
        maximum=1.0,
        hold=5,
    )
    key = jax.random.key(seed + 2)
    for action in actions:
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, action, params)
        state, frame = capture(state)
        frames.append(frame)
    return frames


def _collect_cube(
    *,
    seed: int,
    steps: int,
    image_size: int,
) -> list[np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        env, params = envx.make(
            "cube",
            num_envs=1,
            env_type="single",
            observation_type="pixels",
            image_size=(image_size, image_size),
            task_ids=2,
        )

    observation, state = env.reset(jax.random.key(seed), params)
    frames = [np.asarray(jax.block_until_ready(observation[0]))]
    actions = _held_random_actions(
        jax.random.key(seed + 1),
        steps=steps,
        shape=(1, 5),
        minimum=-1.0,
        maximum=1.0,
        hold=4,
    )
    # Smaller Cartesian/yaw increments keep an uninformed policy in view while
    # retaining the full open/close range for the gripper.
    actions = actions.at[..., :4].multiply(0.4)
    key = jax.random.key(seed + 2)
    for action in actions:
        key, step_key = jax.random.split(key)
        observation, state, _, _, _ = env.step(step_key, state, action, params)
        frames.append(np.asarray(jax.block_until_ready(observation[0])))
    return frames


def _write_rollout(
    name: str,
    *,
    output_dir: Path,
    seed: int,
    steps: int,
    image_size: int,
    frame_duration_ms: int,
) -> None:
    if name in ("pusht", "two-rooms"):
        frames = _collect_pixel_observations(
            name,
            seed=seed,
            steps=steps,
            image_size=image_size,
        )
    elif name == "reacher":
        frames = _collect_reacher(seed=seed, steps=steps, image_size=image_size)
    else:
        frames = _collect_cube(seed=seed, steps=steps, image_size=image_size)

    frames = [
        np.asarray(
            Image.fromarray(frame).resize(
                (image_size, image_size),
                Image.Resampling.LANCZOS,
            )
        )
        for frame in frames
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{name}.gif"
    preview_path = output_dir / f"{name}-preview.png"
    imageio.mimsave(gif_path, frames, duration=frame_duration_ms, loop=0)
    imageio.imwrite(preview_path, frames[len(frames) // 2])
    changed_pixels = np.mean(np.any(frames[0] != frames[-1], axis=-1))
    print(f"{name}: {gif_path} ({len(frames)} frames, {changed_pixels:.1%} pixels changed)")


def _run_all_in_isolated_processes(args: argparse.Namespace) -> None:
    """Keep the two MJX-Warp render contexts in separate processes."""

    script = Path(__file__).resolve()
    for index, name in enumerate(ENVIRONMENTS):
        command = [
            sys.executable,
            str(script),
            name,
            "--output-dir",
            str(args.output_dir),
            "--seed",
            str(args.seed + index),
            "--steps",
            str(args.steps),
            "--image-size",
            str(args.image_size),
            "--frame-duration-ms",
            str(args.frame_duration_ms),
        ]
        subprocess.run(command, check=True)
    _write_grid(args.output_dir, args.image_size, args.frame_duration_ms)


def _write_grid(output_dir: Path, image_size: int, frame_duration_ms: int) -> None:
    """Combine the four equal-length rollouts into one labeled 2x2 GIF."""

    labels = {
        "pusht": "PushT",
        "two-rooms": "Two-Room",
        "reacher": "Reacher",
        "cube": "Cube",
    }
    source_frames = {}
    for name in ENVIRONMENTS:
        path = output_dir / f"{name}.gif"
        source_frames[name] = [
            Image.fromarray(frame).convert("RGB") for frame in imageio.mimread(path)
        ]
    num_frames = min(len(frames) for frames in source_frames.values())
    label_height = 26
    gap = 6
    cell_height = image_size + label_height
    canvas_size = (2 * image_size + gap, 2 * cell_height + gap)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    grid_frames = []
    for frame_index in range(num_frames):
        canvas = Image.new("RGB", canvas_size, (25, 27, 31))
        draw = ImageDraw.Draw(canvas)
        for index, name in enumerate(ENVIRONMENTS):
            column = index % 2
            row = index // 2
            left = column * (image_size + gap)
            top = row * (cell_height + gap)
            frame = source_frames[name][frame_index].resize(
                (image_size, image_size),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(frame, (left, top + label_height))
            draw.text((left + 7, top + 5), labels[name], fill=(245, 245, 245), font=font)
        grid_frames.append(np.asarray(canvas))

    gif_path = output_dir / "all-environments.gif"
    preview_path = output_dir / "all-environments-preview.png"
    imageio.mimsave(gif_path, grid_frames, duration=frame_duration_ms, loop=0)
    imageio.imwrite(preview_path, grid_frames[len(grid_frames) // 2])
    print(f"all: {gif_path} ({len(grid_frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=(*ENVIRONMENTS, "all"))
    parser.add_argument("--output-dir", type=Path, default=Path("rollouts"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--frame-duration-ms", type=int, default=80)
    args = parser.parse_args()
    if args.steps < 1 or args.image_size < 1 or args.frame_duration_ms < 1:
        parser.error("steps, image-size, and frame-duration-ms must be positive")

    if args.environment == "all":
        _run_all_in_isolated_processes(args)
    else:
        _write_rollout(
            args.environment,
            output_dir=args.output_dir,
            seed=args.seed,
            steps=args.steps,
            image_size=args.image_size,
            frame_duration_ms=args.frame_duration_ms,
        )


if __name__ == "__main__":
    main()
