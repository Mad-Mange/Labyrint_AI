import math
import random

import pytest

from labyrint import LabyrinthGame, Status, available_levels, load_level
from labyrint.autopilot import PathFollower, run_episode
from labyrint.level import validate_level


@pytest.mark.parametrize("name", available_levels())
def test_levels_are_valid(name):
    assert validate_level(load_level(name)) == []


def test_flat_board_ball_stays_put():
    game = LabyrinthGame()
    for _ in range(120):
        game.step((0.0, 0.0))
    assert game.ball_pos == game.level.start
    assert game.status is Status.RUNNING


def test_tilt_rolls_ball_in_tilt_direction():
    game = LabyrinthGame()
    x0, y0 = game.ball_pos
    for _ in range(30):
        game.step((0.0, -1.0))   # tilt "up": ball should roll towards smaller y
    assert game.ball_pos[1] < y0 - 5
    assert abs(game.ball_pos[0] - x0) < 1e-9


def test_tilt_follows_target_at_limited_speed():
    game = LabyrinthGame()
    game.step((1.0, 0.0))
    per_tick = game.physics.tilt_speed_deg / game.physics.max_tilt_deg / game.physics.tick_rate
    assert game.tilt[0] == pytest.approx(per_tick)


def test_deterministic():
    def run():
        game = LabyrinthGame(seed=1)
        rng = random.Random(42)
        game.reset(jitter=2.0)
        for _ in range(600):
            game.step((rng.uniform(-1, 1), rng.uniform(-1, 1)))
        return game.state
    assert run() == run()


def test_random_play_never_escapes_or_enters_walls():
    game = LabyrinthGame()
    lv = game.level
    rng = random.Random(0)
    for episode in range(15):
        game.reset(start_s=rng.uniform(0, lv.path.length))
        action = (0.0, 0.0)
        for t in range(900):
            if t % 20 == 0:
                action = (rng.uniform(-1, 1), rng.uniform(-1, 1))
            game.step(action)
            x, y = game.ball_pos
            r = lv.ball_radius
            assert r - 0.5 <= x <= lv.width - r + 0.5 and r - 0.5 <= y <= lv.height - r + 0.5
            assert min(w.signed_distance(x, y) for w in lv.walls) > r - 0.5
            if game.status is not Status.RUNNING:
                break


def test_ball_falls_into_hole():
    game = LabyrinthGame()
    hx, hy = game.level.holes[0]
    game.x, game.y = hx + game.level.hole_radius - 1.0, hy   # centre just over the rim
    for _ in range(60):
        game.step((0.0, 0.0))
    assert game.status is Status.FELL
    assert game.fell_into == 0


def test_fast_ball_can_cross_hole_edge():
    """A ball grazing a hole rim at speed should not be swallowed instantly."""
    game = LabyrinthGame()
    hx, hy = game.level.holes[2]
    R = game.level.hole_radius
    game.x, game.y = hx - 20.0, hy - (R - 1.5)
    game.vx = 500.0
    for _ in range(6):
        game.step((0.0, 0.0))
    assert game.status is Status.RUNNING


def test_progress_tracks_the_line():
    game = LabyrinthGame()
    path = game.level.path
    for s in (100.0, 500.0, 1000.0):
        game.reset(start_s=s)
        assert game.progress_s == pytest.approx(s, abs=0.5)
        assert game.state.holes_passed == sum(1 for hs in game.level.hole_s if hs <= game.progress_s)
    game.reset(start_s=path.length)
    assert game.state.progress == pytest.approx(1.0)


def test_clone_is_independent():
    game = LabyrinthGame()
    game.reset(start_s=300.0)   # top corridor
    for _ in range(20):
        game.step((-0.4, 0.0))
    copy = game.clone()
    before = game.state
    for _ in range(20):
        copy.step((0.4, 0.0))
    assert game.state == before
    assert copy.ball_pos != game.ball_pos
    for _ in range(20):
        game.step((0.4, 0.0))
    assert game.status is Status.RUNNING
    assert copy.state == game.state


def test_autopilot_finishes():
    game = LabyrinthGame(seed=0)
    status, t, holes = run_episode(game, PathFollower(), jitter=2.0)
    assert status is Status.FINISHED, f"autopilot stopped after {holes} holes"
    assert holes == len(game.level.holes)
    assert t < 90


def test_wall_rays_see_frame():
    game = LabyrinthGame()
    x, y = game.level.start
    rays = game.wall_rays(4, 500.0)   # right, down, left, up
    r = game.level.ball_radius
    assert rays[0] == pytest.approx(game.level.width - x - r)
    assert rays[1] == pytest.approx(game.level.height - y - r)
    assert math.isclose(rays[2], x - 213 - r)   # the first inner wall ends at x = 213
