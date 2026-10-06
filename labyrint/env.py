"""Gymnasium environment - the API the neural network will train against.

    import gymnasium as gym
    import labyrint                      # registers "Labyrint-v0"
    env = gym.make("Labyrint-v0", render_mode="human")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Action: 2 floats in [-1, 1] - the target tilt of the board (x, y), exactly what a
human does with the two knobs.

Observation (vector, all values in [-1, 1]):
    ball position (2), ball velocity (2), current tilt (2),
    guide line points 15/40/80 mm ahead relative to the ball (6),
    guide line direction (2), signed distance from the line (1),
    the 4 nearest holes as (proximity, dir x, dir y) (12),
    free distance to walls in 8 directions (8), progress along the line (1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import LabyrinthGame, PhysicsConfig, Status
from .geometry import clamp

LOOKAHEAD_MM = (15.0, 40.0, 80.0)
N_HOLES = 4
N_RAYS = 8
STALL_MM = 5.0   # progress needed to reset the stall timer
STALL_SECONDS = 10.0
OBS_SIZE = 2 + 2 + 2 + 2 * len(LOOKAHEAD_MM) + 2 + 1 + 3 * N_HOLES + N_RAYS + 1


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 0.1        # per mm moved forward along the line (backwards is negative)
    fall: float = -5.0           # dropping into a hole, or stalling (see stall_seconds). Kept mild:
                                 # falling already forfeits all future reward, and a harsh penalty
                                 # teaches the agent to park the ball instead of trying.
    finish: float = 50.0         # reaching FINISH
    time: float = -0.01          # per env step, encourages hurrying
    off_line: float = 0.0        # per step and mm away from the line (0 = off)


def observe(game: LabyrinthGame) -> np.ndarray:
    """The observation vector for a game state (see the module docstring for the layout).

    A module-level function so trained agents can also drive a plain LabyrinthGame (play.py).
    """
    lv = game.level
    x, y = game.ball_pos
    vx, vy = game.ball_vel
    obs = [x / lv.width * 2 - 1, y / lv.height * 2 - 1, vx / 750.0, vy / 750.0, *game.tilt]
    for px, py in game.path_lookahead(LOOKAHEAD_MM):
        obs += [(px - x) / 100.0, (py - y) / 100.0]
    tx, ty = lv.path.tangent_at(game.progress_s)
    qx, qy = lv.path.point_at(game.progress_s)
    cross = (x - qx) * -ty + (y - qy) * tx       # signed offset: + is left of the direction of travel
    obs += [tx, ty, cross / 30.0]
    for dx, dy, d in game.nearest_holes(N_HOLES):
        rim = d - lv.hole_radius
        obs += [1.0 - clamp(rim / 60.0, 0.0, 1.0), dx / max(d, 1e-6), dy / max(d, 1e-6)]
    obs += [r / 60.0 for r in game.wall_rays(N_RAYS, 60.0)]
    obs.append(game.progress_s / lv.path.length * 2 - 1)
    return np.clip(np.asarray(obs, dtype=np.float32), -1.0, 1.0)


class LabyrinthEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, level: str = "classic", render_mode: str | None = None, frame_skip: int = 2,
                 max_episode_seconds: float = 120.0, random_start: float = 0.0, start_jitter: float = 1.0,
                 stall_seconds: float | None = STALL_SECONDS,
                 physics: PhysicsConfig | None = None, reward: RewardConfig | None = None,
                 render_scale: float = 2.6):
        """
        frame_skip: physics ticks (1/60 s each) per env step -> 2 gives 30 decisions per second.
        random_start: probability that an episode starts at a random point on the guide line
            instead of at START. Good for curriculum learning: the agent gets to practise the
            hard final bends without first mastering the whole board.
        stall_seconds: end the episode as a failure (same penalty as a hole) when the ball
            has not advanced STALL_MM along the line for this long. None disables it.
        """
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.game = LabyrinthGame(level, physics)
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_seconds = max_episode_seconds
        self.random_start = random_start
        self.start_jitter = start_jitter
        self.stall_seconds = stall_seconds
        self.rewards = reward or RewardConfig()
        self.render_scale = render_scale
        self.metadata = {**self.metadata, "render_fps": self.game.physics.tick_rate // frame_skip}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self._renderer = None
        self._screen = None
        self._clock = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        start_s = options.get("start_s")
        if start_s is None and self.random_start > 0 and self.np_random.random() < self.random_start:
            start_s = float(self.np_random.uniform(0.0, self.game.level.path.length * 0.97))
        game_seed = int(self.np_random.integers(2**31))
        self.game.reset(start_s=start_s, jitter=self.start_jitter, seed=game_seed)
        if start_s is not None:
            self._nudge_off_holes()
        self._stall_s, self._stall_t = self.game.max_progress_s, 0.0
        self._stalled = False
        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(2)
        game, rw = self.game, self.rewards
        s_before = game.progress_s
        for _ in range(self.frame_skip):
            game.step((float(a[0]), float(a[1])))
            if game.status is not Status.RUNNING:
                break
        reward = rw.progress * (game.progress_s - s_before) + rw.time
        if rw.off_line:
            reward -= rw.off_line * game.level.path.project(*game.ball_pos, game.progress_s - 5,
                                                             game.progress_s + 5)[1]
        if game.max_progress_s >= self._stall_s + STALL_MM:
            self._stall_s, self._stall_t = game.max_progress_s, game.time
        self._stalled = (self.stall_seconds is not None and game.status is Status.RUNNING
                         and game.time - self._stall_t >= self.stall_seconds - 1e-9)
        terminated = game.status is not Status.RUNNING or self._stalled
        if game.status is Status.FELL or self._stalled:
            reward += rw.fall
        elif game.status is Status.FINISHED:
            reward += rw.finish
        truncated = not terminated and game.time >= self.max_episode_seconds
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        import pygame
        from .render import Hud, Renderer
        if self._renderer is None:
            pygame.init()
            panel = self.render_mode == "human"
            self._renderer = Renderer(self.game.level, self.render_scale, panel=panel)
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((self._renderer.width, self._renderer.height))
                pygame.display.set_caption("Labyrint – AI")
                self._clock = pygame.time.Clock()
        if self.render_mode == "rgb_array":
            return self._renderer.board_array(self.game)
        pygame.event.pump()
        self._renderer.draw(self._screen, self.game, Hud(controller="AI"))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self._renderer is not None:
            import pygame
            if self._screen is not None:
                pygame.display.quit()
            self._renderer = self._screen = None

    # ------------------------------------------------------------------ helpers
    def _nudge_off_holes(self) -> None:
        """A random start on the line may land too close to a hole rim - move back a bit."""
        lv, game = self.game.level, self.game
        for _ in range(40):
            _, _, d = game.nearest_holes(1)[0]
            if d > lv.hole_radius + 2.0:
                return
            game.reset(start_s=max(0.0, game.progress_s - 3.0))

    def _observation(self) -> np.ndarray:
        return observe(self.game)

    def _info(self) -> dict:
        st = self.game.state
        return {
            "status": "stalled" if self._stalled else st.status.value,
            "progress": st.progress,
            "holes_passed": st.holes_passed,
            "time": st.time,
            "is_success": st.status is Status.FINISHED,
            "distance_from_line": math.dist(st.ball_pos, self.game.level.path.point_at(st.progress_s)),
        }
