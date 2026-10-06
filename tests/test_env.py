import os

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import labyrint  # noqa: F401  (registers Labyrint-v0)
from labyrint.autopilot import PathFollower
from labyrint.env import OBS_SIZE, LabyrinthEnv


def test_passes_gymnasium_checker():
    check_env(LabyrinthEnv(), skip_render_check=True)


def test_registered_and_shapes():
    env = gym.make("Labyrint-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert {"progress", "holes_passed", "is_success", "status", "time"} <= info.keys()
    env.close()


def test_falling_is_punished_and_terminates():
    env = LabyrinthEnv()
    env.reset(seed=0)
    hx, hy = env.game.level.holes[0]
    env.game.x, env.game.y = hx + env.game.level.hole_radius - 1.0, hy   # just over the rim of hole 1
    for _ in range(60):
        _, r, terminated, truncated, info = env.step(np.zeros(2))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["status"] == "fell"
    assert r <= env.rewards.fall + 1.0


def test_parking_the_ball_counts_as_failure():
    env = LabyrinthEnv(stall_seconds=5.0)
    env.reset(seed=0)
    steps = 0
    while True:   # START is in a corner: tilting into it keeps the ball still
        _, r, terminated, truncated, info = env.step(np.ones(2))
        steps += 1
        if terminated or truncated:
            break
    assert terminated and info["status"] == "stalled" and not info["is_success"]
    assert r <= env.rewards.fall + 1.0
    assert steps == round(5.0 * 30)


def test_autopilot_through_env_gets_success():
    env = LabyrinthEnv()
    env.reset(seed=3)
    pilot = PathFollower()
    total = 0.0
    while True:
        _, r, terminated, truncated, info = env.step(np.array(pilot.act(env.game)))
        total += r
        if terminated or truncated:
            break
    assert info["is_success"]
    assert total > env.rewards.finish


def test_random_start_curriculum():
    env = LabyrinthEnv(random_start=1.0)
    starts = [env.reset(seed=i)[1]["progress"] for i in range(20)]
    assert max(starts) > 0.3 and len(set(round(s, 3) for s in starts)) > 10


def test_rgb_array_render():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = LabyrinthEnv(render_mode="rgb_array", render_scale=1.0)
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    assert frame.std() > 5   # not a blank image
    env.close()
