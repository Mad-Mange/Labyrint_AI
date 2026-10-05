"""Geometry primitives: oriented wall rectangles, the guide-line path and a spatial grid.

All coordinates are in millimetres on the board surface, x to the right and y downwards
(the same orientation as the screen).
"""
from __future__ import annotations

import bisect
import math
from typing import Iterable, Sequence

Point = tuple[float, float]


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Wall:
    """An oriented rectangle - one of the wooden strips glued onto the board."""

    __slots__ = ("cx", "cy", "hw", "hh", "angle", "cos", "sin", "corners", "aabb")

    def __init__(self, cx: float, cy: float, hw: float, hh: float, angle: float = 0.0):
        self.cx, self.cy = float(cx), float(cy)
        self.hw, self.hh = float(hw), float(hh)
        self.angle = float(angle)
        self.cos, self.sin = math.cos(angle), math.sin(angle)
        self.corners = [self.to_world(sx * self.hw, sy * self.hh)
                        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        self.aabb = (min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def from_rect(cls, x: float, y: float, w: float, h: float) -> "Wall":
        return cls(x + w / 2, y + h / 2, w / 2, h / 2)

    @classmethod
    def from_segment(cls, x1: float, y1: float, x2: float, y2: float, thickness: float) -> "Wall":
        dx, dy = x2 - x1, y2 - y1
        return cls((x1 + x2) / 2, (y1 + y2) / 2, math.hypot(dx, dy) / 2, thickness / 2,
                   math.atan2(dy, dx))

    def to_local(self, px: float, py: float) -> Point:
        dx, dy = px - self.cx, py - self.cy
        return dx * self.cos + dy * self.sin, -dx * self.sin + dy * self.cos

    def to_world(self, lx: float, ly: float) -> Point:
        return self.cx + lx * self.cos - ly * self.sin, self.cy + lx * self.sin + ly * self.cos

    def rotate(self, lx: float, ly: float) -> Point:
        """Local direction -> world direction."""
        return lx * self.cos - ly * self.sin, lx * self.sin + ly * self.cos

    def signed_distance(self, px: float, py: float) -> float:
        """Distance from a point to the rectangle surface (negative inside)."""
        lx, ly = self.to_local(px, py)
        qx, qy = abs(lx) - self.hw, abs(ly) - self.hh
        return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0)

    def raycast(self, ox: float, oy: float, dx: float, dy: float, max_t: float) -> float:
        """Distance along the unit ray (dx, dy) until it enters the rectangle, or max_t."""
        lox, loy = self.to_local(ox, oy)
        ldx = dx * self.cos + dy * self.sin
        ldy = -dx * self.sin + dy * self.cos
        t0, t1 = 0.0, max_t
        for o, d, h in ((lox, ldx, self.hw), (loy, ldy, self.hh)):
            if abs(d) < 1e-12:
                if o < -h or o > h:
                    return max_t
            else:
                ta, tb = (-h - o) / d, (h - o) / d
                if ta > tb:
                    ta, tb = tb, ta
                t0 = max(t0, ta)
                t1 = min(t1, tb)
                if t0 > t1:
                    return max_t
        return t0


def chaikin(points: Sequence[Point], iterations: int) -> list[Point]:
    """Chaikin corner cutting; keeps the end points fixed."""
    pts = [tuple(map(float, p)) for p in points]
    for _ in range(iterations):
        new = [pts[0]]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            new.append((0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by))
            new.append((0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by))
        new.append(pts[-1])
        pts = new
    return pts


class Path:
    """The black guide line from START to FINISH, parameterised by arc length s (mm)."""

    def __init__(self, points: Sequence[Point]):
        pts: list[Point] = []
        for p in points:
            p = (float(p[0]), float(p[1]))
            if not pts or math.dist(pts[-1], p) > 1e-9:
                pts.append(p)
        if len(pts) < 2:
            raise ValueError("a path needs at least two distinct points")
        self.points = pts
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.length = self.cum[-1]

    def _segment(self, s: float) -> int:
        i = bisect.bisect_right(self.cum, s) - 1
        return min(max(i, 0), len(self.points) - 2)

    def point_at(self, s: float) -> Point:
        s = clamp(s, 0.0, self.length)
        i = self._segment(s)
        (ax, ay), (bx, by) = self.points[i], self.points[i + 1]
        seg = self.cum[i + 1] - self.cum[i]
        t = (s - self.cum[i]) / seg if seg > 0 else 0.0
        return ax + (bx - ax) * t, ay + (by - ay) * t

    def tangent_at(self, s: float) -> Point:
        i = self._segment(clamp(s, 0.0, self.length))
        (ax, ay), (bx, by) = self.points[i], self.points[i + 1]
        d = math.hypot(bx - ax, by - ay)
        return (bx - ax) / d, (by - ay) / d

    def project(self, px: float, py: float, s_lo: float = 0.0,
                s_hi: float = math.inf) -> tuple[float, float, Point]:
        """Closest point on the path among the segments overlapping [s_lo, s_hi].

        Returns (s, distance, closest_point). Restricting the window keeps the
        progress tracker from jumping to a neighbouring lap of the spiral.
        """
        i0 = self._segment(max(s_lo, 0.0))
        i1 = self._segment(min(s_hi, self.length))
        best_d2, best_s, best_q = math.inf, 0.0, self.points[0]
        for i in range(i0, i1 + 1):
            (ax, ay), (bx, by) = self.points[i], self.points[i + 1]
            abx, aby = bx - ax, by - ay
            l2 = abx * abx + aby * aby
            t = clamp(((px - ax) * abx + (py - ay) * aby) / l2, 0.0, 1.0) if l2 > 0 else 0.0
            qx, qy = ax + abx * t, ay + aby * t
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = self.cum[i] + t * (self.cum[i + 1] - self.cum[i])
                best_q = (qx, qy)
        return best_s, math.sqrt(best_d2), best_q

    def sample(self, step: float) -> Iterable[Point]:
        n = max(1, int(math.ceil(self.length / step)))
        for k in range(n + 1):
            yield self.point_at(self.length * k / n)


class SpatialGrid:
    """Uniform grid for broad-phase lookups: which walls/holes are near a point."""

    def __init__(self, cell: float):
        self.cell = float(cell)
        self._cells: dict[tuple[int, int], list[int]] = {}

    def insert(self, index: int, x0: float, y0: float, x1: float, y1: float) -> None:
        c = self.cell
        for ix in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
            for iy in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                self._cells.setdefault((ix, iy), []).append(index)

    def query(self, x: float, y: float) -> Sequence[int]:
        return self._cells.get((int(math.floor(x / self.cell)), int(math.floor(y / self.cell))), ())
