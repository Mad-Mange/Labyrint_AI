"""A hand-written controller that follows the guide line - no learning involved.

It proves that a level is solvable with the engine's physics and gives a baseline
time for the neural network to beat.

    python -m labyrint.autopilot --episodes 20
"""
from __future__ import annotations

import argparse
import math
import time

from .game import LabyrinthGame, Status
from .geometry import clamp

BASELINE_SECONDS = 38.0   # what PathFollower takes on "classic" - the time for the AI to beat


class PathFollower:
    def __init__(self, speed: float = 120.0, gain: float = 5.0, track_gain: float = 4.0,
                 corner_lookahead: float = 30.0, caution_dist: float = 22.0):
        self.speed = speed                    # cruising speed along the line (mm/s)
        self.gain = gain                      # 1/s: how hard to correct velocity errors
        self.track_gain = track_gain          # 1/s: how hard to pull back onto the line
        self.corner_lookahead = corner_lookahead  # brake for bends this far ahead (mm)
        self.caution_dist = caution_dist      # slow down when a hole rim is closer than this (mm)

    def act(self, game: LabyrinthGame) -> tuple[float, float]:
        path = game.level.path
        s = game.progress_s
        px, py = game.ball_pos
        vx, vy = game.ball_vel
        qx, qy = path.point_at(s)
        t1x, t1y = path.tangent_at(s + 4.0)
        t2x, t2y = path.tangent_at(s + self.corner_lookahead)
        bend = math.acos(clamp(t1x * t2x + t1y * t2y, -1.0, 1.0))

        speed = self.speed * clamp(1.0 - bend / 1.4, 0.3, 1.0)
        (_, _, hole_d), = game.nearest_holes(1)
        speed *= clamp((hole_d - game.level.hole_radius) / self.caution_dist, 0.35, 1.0)
        if s >= path.length - 1.0:
            speed = 0.0

        want_vx = t1x * speed + self.track_gain * (qx - px)
        want_vy = t1y * speed + self.track_gain * (qy - py)
        ax = self.gain * (want_vx - vx)
        ay = self.gain * (want_vy - vy)
        return clamp(ax / game.max_accel, -1, 1), clamp(ay / game.max_accel, -1, 1)


def run_episode(game: LabyrinthGame, pilot: PathFollower, max_seconds: float = 120.0,
                **reset_kwargs) -> tuple[Status, float, int]:
    game.reset(**reset_kwargs)
    while game.status is Status.RUNNING and game.time < max_seconds:
        game.step(pilot.act(game))
    return game.status, game.time, game.state.holes_passed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the path-following autopilot headless.")
    ap.add_argument("--level", default="classic")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=2.0, help="random start offset (mm)")
    args = ap.parse_args()

    game = LabyrinthGame(args.level, seed=0)
    pilot = PathFollower()
    wins, ticks = 0, 0
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        status, t, holes = run_episode(game, pilot, jitter=args.jitter)
        ticks += round(t * game.physics.tick_rate)
        wins += status is Status.FINISHED
        extra = "" if status is Status.FINISHED else f" (hole {game.fell_into + 1 if game.fell_into is not None else '-'})"
        print(f"episode {ep + 1:3d}: {status.value:8s} {t:6.1f} s  passed {holes:2d} holes{extra}")
    wall = time.perf_counter() - t0
    print(f"\n{wins}/{args.episodes} finished. Simulated {ticks / game.physics.tick_rate:.0f} s "
          f"in {wall:.1f} s ({ticks / wall:.0f} ticks/s, {ticks / wall / game.physics.tick_rate:.0f}x real time)")


if __name__ == "__main__":
    main()
