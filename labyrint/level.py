"""Level definitions: board size, wooden walls, holes and the guide line.

Levels live as JSON files in ``labyrint/levels``. Format::

    {
      "name": "Klassisk",
      "size": [260, 260],                       # playing surface in mm
      "ball_radius": 6, "hole_radius": 8, "wall_thickness": 8,
      "start": [x, y], "finish": [x, y], "finish_radius": 9,
      "walls": [{"rect": [x, y, w, h]}, {"seg": [x1, y1, x2, y2]}],
      "holes": [[x, y], ...],                   # in path order -> numbered 1..N
      "path": [[x, y], ...],                    # control points of the black line
      "path_smoothing": 3                       # Chaikin iterations
    }
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path as FsPath

from .geometry import Path, Point, Wall, chaikin

LEVEL_DIR = FsPath(__file__).parent / "levels"
FRAME_THICKNESS = 50.0  # invisible collision slabs just outside the playing surface


@dataclass
class Level:
    name: str
    width: float
    height: float
    ball_radius: float
    hole_radius: float
    wall_thickness: float
    start: Point
    finish: Point
    finish_radius: float
    walls: list[Wall]
    holes: list[Point]
    path: Path
    hole_s: list[float] = field(default_factory=list)
    frame: list[Wall] = field(default_factory=list)

    def __post_init__(self) -> None:
        w, h, t = self.width, self.height, FRAME_THICKNESS
        self.frame = [
            Wall.from_rect(-t, -t, w + 2 * t, t),  # top
            Wall.from_rect(-t, h, w + 2 * t, t),   # bottom
            Wall.from_rect(-t, 0, t, h),           # left
            Wall.from_rect(w, 0, t, h),            # right
        ]
        self.hole_s = [self.path.project(x, y)[0] for x, y in self.holes]

    @property
    def all_walls(self) -> list[Wall]:
        return self.walls + self.frame

    def holes_passed(self, s: float) -> int:
        """How many numbered holes lie behind arc length s - the classic score."""
        return sum(1 for hs in self.hole_s if hs <= s)


def level_from_dict(d: dict) -> Level:
    thickness = float(d.get("wall_thickness", 8))
    walls = []
    for spec in d["walls"]:
        if "rect" in spec:
            walls.append(Wall.from_rect(*spec["rect"]))
        elif "seg" in spec:
            walls.append(Wall.from_segment(*spec["seg"], spec.get("thickness", thickness)))
        else:
            raise ValueError(f"unknown wall spec: {spec}")
    path_points = chaikin([tuple(p) for p in d["path"]], int(d.get("path_smoothing", 3)))
    return Level(
        name=d.get("name", "Namnlös"),
        width=float(d["size"][0]),
        height=float(d["size"][1]),
        ball_radius=float(d.get("ball_radius", 6)),
        hole_radius=float(d.get("hole_radius", 8)),
        wall_thickness=thickness,
        start=tuple(map(float, d["start"])),
        finish=tuple(map(float, d["finish"])),
        finish_radius=float(d.get("finish_radius", 9)),
        walls=walls,
        holes=[tuple(map(float, h)) for h in d["holes"]],
        path=Path(path_points),
    )


def available_levels() -> list[str]:
    return sorted(p.stem for p in LEVEL_DIR.glob("*.json"))


def load_level(name_or_path: str | FsPath = "classic") -> Level:
    p = FsPath(name_or_path)
    if not p.suffix:
        p = LEVEL_DIR / f"{name_or_path}.json"
    with open(p, encoding="utf-8") as f:
        return level_from_dict(json.load(f))


def validate_level(level: Level, hole_clearance: float = 4.0) -> list[str]:
    """Sanity checks so that a level is actually solvable along its guide line.

    hole_clearance: how far (mm) the line must stay outside each hole's rim.
    """
    problems = []
    r, R = level.ball_radius, level.hole_radius
    walls = level.all_walls

    for i, (hx, hy) in enumerate(level.holes, 1):
        if not (R <= hx <= level.width - R and R <= hy <= level.height - R):
            problems.append(f"hole {i} at {(hx, hy)} sticks out of the board")
        for w in level.walls:
            if w.signed_distance(hx, hy) < R - 0.5:
                problems.append(f"hole {i} at {(hx, hy)} overlaps a wall")
        for j, (ox, oy) in enumerate(level.holes[i:], i + 1):
            if math.hypot(hx - ox, hy - oy) < 2 * R + 2:
                problems.append(f"holes {i} and {j} overlap")

    for name, (px, py) in (("start", level.start), ("finish", level.finish)):
        if min(w.signed_distance(px, py) for w in walls) < r:
            problems.append(f"{name} is inside or touching a wall")
        if any(math.hypot(px - hx, py - hy) < R + r for hx, hy in level.holes):
            problems.append(f"{name} is too close to a hole")

    for px, py in level.path.sample(1.0):
        dw = min(w.signed_distance(px, py) for w in walls)
        if dw < r + 0.5:
            problems.append(f"path at ({px:.1f}, {py:.1f}) is {dw:.1f} mm from a wall")
        for i, (hx, hy) in enumerate(level.holes, 1):
            dh = math.hypot(px - hx, py - hy)
            if dh < R + hole_clearance:
                problems.append(f"path at ({px:.1f}, {py:.1f}) is {dh:.1f} mm from hole {i}")

    if math.dist(level.path.points[0], level.start) > 1e-6:
        problems.append("path does not begin at start")
    if math.dist(level.path.points[-1], level.finish) > 1e-6:
        problems.append("path does not end at finish")
    if any(b < a for a, b in zip(level.hole_s, level.hole_s[1:])):
        problems.append("holes are not listed in path order")
    return problems
