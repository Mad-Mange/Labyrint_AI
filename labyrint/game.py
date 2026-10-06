"""Headless game engine: a steel ball rolling on a tiltable wooden board.

The engine knows nothing about graphics or input, so it can be stepped far faster
than real time - which is what training an AI needs. Units are mm, seconds, and
tilt is normalised to [-1, 1] per axis (1 == max_tilt_deg).

    game = LabyrinthGame("classic")
    game.reset()
    while game.status is Status.RUNNING:
        game.step((0.3, -0.5))       # target tilt: + x rolls the ball right, + y rolls it down
"""
from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass

from .geometry import Point, SpatialGrid, clamp
from .level import Level, load_level

ROLLING_FACTOR = 5.0 / 7.0  # a solid sphere rolling without slipping: a = 5/7 * g * sin(angle)


class Status(enum.Enum):
    RUNNING = "running"
    FELL = "fell"          # dropped into a hole
    FINISHED = "finished"  # reached the finish


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float = 9810.0          # mm/s^2
    max_tilt_deg: float = 3.0        # how far the knobs can tilt the board
    tilt_speed_deg: float = 25.0     # how fast the board can change its tilt (deg/s)
    rolling_resistance: float = 0.004
    restitution: float = 0.35        # bounciness against the wooden walls
    wall_friction: float = 0.08      # fraction of tangential speed lost per wall hit
    hole_pull: float = 1.0           # strength of the rim tipping the ball into a hole
    fall_fraction: float = 0.5       # ball is gone once its centre is this * hole_radius from the hole centre
    max_speed: float = 1500.0        # mm/s
    tick_rate: int = 60              # step() advances 1 / tick_rate seconds
    substeps: int = 4                # physics substeps per tick


@dataclass(frozen=True)
class GameState:
    """An immutable snapshot of everything that changes during play."""
    ball_pos: Point
    ball_vel: Point
    tilt: Point
    target_tilt: Point
    time: float
    status: Status
    progress_s: float       # arc length along the guide line (mm)
    max_progress_s: float
    progress: float         # progress_s / path length, 0..1
    holes_passed: int       # the classic score: number of numbered holes behind the ball
    ball_depth: float       # 0 on the board, ->1 as the ball tips into a hole (for rendering)
    fell_into: int | None   # index of the hole the ball fell into
    wall_impact: float      # strongest wall impact speed during the last tick (mm/s)


class LabyrinthGame:
    def __init__(self, level: Level | str = "classic", physics: PhysicsConfig | None = None,
                 seed: int | None = None):
        self.level = load_level(level) if isinstance(level, str) else level
        self.physics = physics or PhysicsConfig()
        self.rng = random.Random(seed)
        self.max_accel = ROLLING_FACTOR * self.physics.gravity * math.sin(
            math.radians(self.physics.max_tilt_deg))

        lv = self.level
        margin = lv.ball_radius + 2.0
        self._walls = lv.all_walls
        self._wall_grid = SpatialGrid(20.0)
        for i, w in enumerate(self._walls):
            x0, y0, x1, y1 = w.aabb
            self._wall_grid.insert(i, max(x0, -10) - margin, max(y0, -10) - margin,
                                   min(x1, lv.width + 10) + margin, min(y1, lv.height + 10) + margin)
        self._hole_grid = SpatialGrid(20.0)
        R = lv.hole_radius
        for i, (hx, hy) in enumerate(lv.holes):
            self._hole_grid.insert(i, hx - R, hy - R, hx + R, hy + R)
        self.reset()

    # ------------------------------------------------------------------ control
    def reset(self, *, start_s: float | None = None, jitter: float = 0.0,
              seed: int | None = None) -> GameState:
        """Put the ball back at START (or at arc length start_s on the guide line).

        jitter: random offset (mm) of the start position - useful for AI training.
        """
        if seed is not None:
            self.rng.seed(seed)
        path = self.level.path
        if start_s is None:
            self.x, self.y = self.level.start
            s = 0.0
        else:
            s = clamp(float(start_s), 0.0, path.length)
            self.x, self.y = path.point_at(s)
        if jitter > 0:
            self.x += self.rng.uniform(-jitter, jitter)
            self.y += self.rng.uniform(-jitter, jitter)
        self.vx = self.vy = 0.0
        self.tx = self.ty = 0.0
        self.target = (0.0, 0.0)
        self.time = 0.0
        self.status = Status.RUNNING
        self.progress_s = s
        self.max_progress_s = s
        self.ball_depth = 0.0
        self.fell_into: int | None = None
        self.wall_impact = 0.0
        self._update_progress()
        return self.state

    def step(self, target_tilt: tuple[float, float] = (0.0, 0.0)) -> GameState:
        """Advance one tick. target_tilt is where the knobs want the board, each axis in [-1, 1]."""
        if self.status is not Status.RUNNING:
            return self.state
        ph = self.physics
        self.target = (clamp(float(target_tilt[0]), -1.0, 1.0), clamp(float(target_tilt[1]), -1.0, 1.0))
        dt = 1.0 / (ph.tick_rate * ph.substeps)
        max_dtilt = ph.tilt_speed_deg / ph.max_tilt_deg * dt
        self.wall_impact = 0.0
        for _ in range(ph.substeps):
            self.tx += clamp(self.target[0] - self.tx, -max_dtilt, max_dtilt)
            self.ty += clamp(self.target[1] - self.ty, -max_dtilt, max_dtilt)
            self._substep(dt)
            if self.status is not Status.RUNNING:
                break
        self.time += 1.0 / ph.tick_rate
        self._update_progress()
        return self.state

    # ------------------------------------------------------------------ physics
    def _substep(self, dt: float) -> None:
        ph, lv = self.physics, self.level
        g = ph.gravity
        theta = math.radians(ph.max_tilt_deg)
        ax = ROLLING_FACTOR * g * math.sin(self.tx * theta)
        ay = ROLLING_FACTOR * g * math.sin(self.ty * theta)

        # Over a hole the ball rests on the rim and tips towards the hole centre.
        r, R = lv.ball_radius, lv.hole_radius
        depth = 0.0
        for i in self._hole_grid.query(self.x, self.y):
            hx, hy = lv.holes[i]
            dx, dy = hx - self.x, hy - self.y
            d = math.hypot(dx, dy)
            if d < R:
                tip = min(1.0, (R - d) / r)
                if d > 1e-9:
                    pull = ROLLING_FACTOR * g * tip * ph.hole_pull
                    ax += pull * dx / d
                    ay += pull * dy / d
                depth = max(depth, (R - d) / (R * (1.0 - ph.fall_fraction)))
        self.ball_depth = min(depth, 1.0)

        vx, vy = self.vx + ax * dt, self.vy + ay * dt
        speed = math.hypot(vx, vy)
        decel = ph.rolling_resistance * g * dt
        if speed <= decel:
            vx = vy = 0.0
        else:
            k = (speed - decel) / speed
            if speed - decel > ph.max_speed:
                k = ph.max_speed / speed
            vx *= k
            vy *= k
        self.vx, self.vy = vx, vy
        self.x += vx * dt
        self.y += vy * dt

        for _ in range(2):  # a second pass settles corner contacts
            for i in self._wall_grid.query(self.x, self.y):
                self._collide(self._walls[i], r)

        for i in self._hole_grid.query(self.x, self.y):
            hx, hy = lv.holes[i]
            if math.hypot(hx - self.x, hy - self.y) < R * ph.fall_fraction:
                self.status = Status.FELL
                self.fell_into = i
                self.ball_depth = 1.0
                self.vx = self.vy = 0.0
                return

        fx, fy = lv.finish
        if math.hypot(fx - self.x, fy - self.y) < lv.finish_radius:
            self.status = Status.FINISHED

    def _collide(self, w, r: float) -> None:
        lx, ly = w.to_local(self.x, self.y)
        if abs(lx) >= w.hw + r or abs(ly) >= w.hh + r:
            return
        qx, qy = clamp(lx, -w.hw, w.hw), clamp(ly, -w.hh, w.hh)
        ddx, ddy = lx - qx, ly - qy
        d2 = ddx * ddx + ddy * ddy
        if d2 > 1e-12:
            if d2 >= r * r:
                return
            d = math.sqrt(d2)
            nlx, nly, pen = ddx / d, ddy / d, r - d
        else:  # centre inside the strip: push out along the shallowest axis
            pen_x, pen_y = w.hw - abs(lx), w.hh - abs(ly)
            if pen_x < pen_y:
                nlx, nly, pen = (1.0 if lx >= 0 else -1.0), 0.0, pen_x + r
            else:
                nlx, nly, pen = 0.0, (1.0 if ly >= 0 else -1.0), pen_y + r
        nx, ny = w.rotate(nlx, nly)
        self.x += nx * pen
        self.y += ny * pen
        vn = self.vx * nx + self.vy * ny
        if vn < 0:
            ph = self.physics
            tvx, tvy = self.vx - vn * nx, self.vy - vn * ny
            keep = 1.0 - ph.wall_friction
            self.vx = tvx * keep - ph.restitution * vn * nx
            self.vy = tvy * keep - ph.restitution * vn * ny
            self.wall_impact = max(self.wall_impact, -vn)

    def _update_progress(self) -> None:
        # Search only near the previous position on the line so progress can't
        # jump through a wall to a neighbouring lap of the spiral.
        s, _, _ = self.level.path.project(self.x, self.y, self.progress_s - 60.0, self.progress_s + 60.0)
        self.progress_s = s
        self.max_progress_s = max(self.max_progress_s, s)

    # ------------------------------------------------------------------ queries
    @property
    def ball_pos(self) -> Point:
        return self.x, self.y

    @property
    def ball_vel(self) -> Point:
        return self.vx, self.vy

    @property
    def tilt(self) -> Point:
        return self.tx, self.ty

    @property
    def state(self) -> GameState:
        lv = self.level
        return GameState(
            ball_pos=(self.x, self.y), ball_vel=(self.vx, self.vy), tilt=(self.tx, self.ty),
            target_tilt=self.target, time=self.time, status=self.status,
            progress_s=self.progress_s, max_progress_s=self.max_progress_s,
            progress=self.progress_s / lv.path.length,
            holes_passed=(len(lv.holes) if self.status is Status.FINISHED
                          else lv.holes_passed(self.max_progress_s)),
            ball_depth=self.ball_depth, fell_into=self.fell_into, wall_impact=self.wall_impact)

    def clone(self) -> "LabyrinthGame":
        """Cheap copy of the dynamic state (the level is shared) - handy for look-ahead search."""
        g = object.__new__(LabyrinthGame)
        g.__dict__.update(self.__dict__)
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        return g

    # ------------------------------------------------------------------ sensors (for AI)
    def path_lookahead(self, distances: tuple[float, ...]) -> list[Point]:
        """Points on the guide line `distances` mm ahead of the ball's current progress."""
        path = self.level.path
        return [path.point_at(self.progress_s + d) for d in distances]

    def nearest_holes(self, k: int) -> list[tuple[float, float, float]]:
        """The k nearest holes as (dx, dy, distance) from the ball centre."""
        out = sorted(((hx - self.x, hy - self.y, math.hypot(hx - self.x, hy - self.y))
                      for hx, hy in self.level.holes), key=lambda t: t[2])
        return out[:k]

    def wall_rays(self, n: int = 8, max_dist: float = 60.0) -> list[float]:
        """Free distance (mm, from the ball's surface) to the nearest wall in n directions."""
        r = self.level.ball_radius
        reach = max_dist + r
        x0, y0, x1, y1 = self.x - reach, self.y - reach, self.x + reach, self.y + reach
        # Only walls whose bounding box is within reach can be hit (same result, ~5x faster).
        near = [w for w in self._walls if w.aabb[0] <= x1 and w.aabb[2] >= x0
                and w.aabb[1] <= y1 and w.aabb[3] >= y0]
        out = []
        for k in range(n):
            a = 2.0 * math.pi * k / n
            dx, dy = math.cos(a), math.sin(a)
            t = reach
            for w in near:
                t = min(t, w.raycast(self.x, self.y, dx, dy, t))
            out.append(max(0.0, t - r))
        return out
