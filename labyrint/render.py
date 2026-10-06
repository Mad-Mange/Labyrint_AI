"""Pygame renderer: a top-down view of the wooden board plus a side panel.

The renderer only *reads* a LabyrinthGame - it never changes it - so the same
code draws a human game, the autopilot, or a neural network playing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame
import pygame.gfxdraw

from .game import LabyrinthGame, Status
from .lang import t
from .level import Level

LIGHT = (-0.6, -0.8)  # light comes from the top left, shadows fall to the bottom right

BOARD_BASE, BOARD_DARK = (238, 214, 172), (204, 164, 112)
FRAME_BASE, FRAME_DARK = (214, 172, 116), (158, 110, 62)
WALL_BASE, WALL_DARK = (226, 190, 136), (182, 138, 84)
INK = (28, 22, 18)
PANEL_BG = (36, 29, 24)
PANEL_TEXT = (240, 226, 200)
PANEL_DIM = (170, 150, 125)
ACCENT = (232, 176, 84)
GOOD = (130, 200, 120)
BAD = (230, 110, 90)


@dataclass
class Hud:
    """Extra information for the side panel that the game engine doesn't track."""
    controller: str = "Mus"
    attempts: int = 1
    best_time: float | None = None
    best_holes: int = 0
    message: str = ""
    message_color: tuple[int, int, int] = PANEL_TEXT
    paused: bool = False
    info: str = ""          # small extra lines under "Styrs av" (e.g. training progress)


# --------------------------------------------------------------------------- textures
def _value_noise(w: int, h: int, cx: int, cy: int, rng: np.random.Generator) -> np.ndarray:
    grid = rng.random((cy + 2, cx + 2))
    ys = np.linspace(0, cy, h, endpoint=False)
    xs = np.linspace(0, cx, w, endpoint=False)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x0 + 1] * fx
    bot = grid[y0 + 1][:, x0] * (1 - fx) + grid[y0 + 1][:, x0 + 1] * fx
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def wood_texture(w: int, h: int, base, dark, period: float, seed: int) -> pygame.Surface:
    """Procedural pine: wavy growth rings running along x."""
    rng = np.random.default_rng(seed)
    y = np.arange(h, dtype=float)[:, None]
    warp = (_value_noise(w, h, 2, 3, rng) * period * 1.6
            + _value_noise(w, h, 7, 9, rng) * period * 0.35)
    phase = (y + warp) / period + rng.random()
    rings = (0.5 + 0.5 * np.cos(2 * np.pi * phase)) ** 6
    streaks = _value_noise(w, h, max(2, w // 90), max(2, h // 2), rng)
    blotch = _value_noise(w, h, 3, 3, rng)
    mix = np.clip(0.55 * rings + 0.22 * streaks + 0.18 * blotch, 0, 1)[..., None]
    rgb = np.array(base, float) * (1 - mix) + np.array(dark, float) * mix
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2).astype(np.uint8))


def _rgba_surface(rgb: np.ndarray, alpha: np.ndarray) -> pygame.Surface:
    h, w = alpha.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    px = pygame.surfarray.pixels3d(surf)
    px[:] = np.clip(rgb, 0, 255).transpose(1, 0, 2).astype(np.uint8)
    del px
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:] = np.clip(alpha, 0, 255).T.astype(np.uint8)
    del pa
    return surf


def _disc_coords(radius_px: float, pad: float = 1.0):
    size = int(math.ceil(2 * (radius_px + pad))) + 2
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(float) + 0.5
    return size, (xx - c), (yy - c)


def ball_sprite(radius_px: float) -> pygame.Surface:
    """A polished steel ball reflecting the room above and the wooden board at its rim."""
    _, dx, dy = _disc_coords(radius_px)
    nx, ny = dx / radius_px, dy / radius_px
    rr = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1 - rr, 0, 1))
    rz = 2 * nz * nz - 1                       # z of the reflected view ray
    center = np.array([205, 210, 222], float)
    mid = np.array([72, 74, 84], float)
    rim = np.array([196, 156, 108], float)
    up = np.clip(rz, 0, 1)[..., None] ** 0.8
    down = (np.clip(-rz, 0, 1)[..., None] ** 1.5) * 0.85
    env = np.where(rz[..., None] > 0, mid * (1 - up) + center * up, mid * (1 - down) + rim * down)
    lx, ly, lz = -0.45, -0.6, 0.66
    norm = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / norm, ly / norm, lz / norm
    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)[..., None]
    rx, ry = 2 * nz * nx, 2 * nz * ny
    spec = np.clip(rx * lx + ry * ly + rz * lz, 0, 1) ** 40
    rgb = env * (0.6 + 0.4 * diffuse) + spec[..., None] * 255
    alpha = np.clip(radius_px - np.sqrt(dx * dx + dy * dy) + 0.5, 0, 1) * 255
    return _rgba_surface(rgb, alpha)


def shadow_sprite(radius_px: float) -> pygame.Surface:
    _, dx, dy = _disc_coords(radius_px * 1.3, pad=2)
    d = np.sqrt(dx * dx + dy * dy) / (radius_px * 1.15)
    alpha = np.clip(1 - d, 0, 1) ** 1.6 * 120
    return _rgba_surface(np.zeros(alpha.shape + (3,)), alpha)


def hole_sprite(radius_px: float, rim_px: float) -> pygame.Surface:
    """A drilled hole: dark inside, far inner wall catching the light, worn rim."""
    _, dx, dy = _disc_coords(radius_px + rim_px)
    d = np.sqrt(dx * dx + dy * dy)
    inside = d < radius_px
    ux, uy = dx / np.maximum(d, 1e-6), dy / np.maximum(d, 1e-6)
    far = np.clip(-(ux * LIGHT[0] + uy * LIGHT[1]) + 0.15, 0, 1) ** 1.3  # inner wall facing the light
    band = np.clip((d / radius_px - 0.4) / 0.6, 0, 1) ** 1.6
    deep = np.array([18, 12, 8], float)
    wall = np.array([176, 128, 78], float)
    rgb = deep + (wall - deep) * (far * band * 0.9)[..., None]
    rim_t = np.clip((d - radius_px) / rim_px, 0, 1)
    rim_rgb = np.array([160, 118, 72], float)
    rgb = np.where(inside[..., None], rgb, np.broadcast_to(rim_rgb, rgb.shape))
    alpha_in = np.clip(radius_px - d + 0.5, 0, 1)
    alpha_rim = (1 - rim_t) ** 1.5 * 0.55
    alpha = np.maximum(alpha_in, np.where(d <= radius_px + rim_px, alpha_rim, 0)) * 255
    return _rgba_surface(rgb, alpha)


def _blur(surface: pygame.Surface, radius: int) -> pygame.Surface:
    try:
        return pygame.transform.gaussian_blur(surface, radius)
    except (AttributeError, ValueError):
        w, h = surface.get_size()
        small = pygame.transform.smoothscale(surface, (max(1, w // 4), max(1, h // 4)))
        return pygame.transform.smoothscale(small, (w, h))


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("segoeui,arial,dejavusans", size, bold=bold)


# --------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, level: Level, scale: float = 2.6, panel: bool = True, frame_mm: float = 16.0):
        pygame.font.init()
        self.level = level
        self.scale = s = scale
        self.frame = round(frame_mm * s)
        self.board_w, self.board_h = round(level.width * s), round(level.height * s)
        self.board_rect = pygame.Rect(self.frame, self.frame, self.board_w, self.board_h)
        self.outer_w = self.board_w + 2 * self.frame
        self.outer_h = self.board_h + 2 * self.frame
        self.panel_w = 300 if panel else 0
        self.width, self.height = self.outer_w + self.panel_w, self.outer_h

        self.ball = ball_sprite(level.ball_radius * s)
        self.ball_shadow = shadow_sprite(level.ball_radius * s)
        self.static = self._build_static()
        self.tilt_shades = self._build_tilt_shades()
        if panel:
            self.f_title = _font(40, bold=True)
            self.f_big = _font(24, bold=True)
            self.f_text = _font(19)
            self.f_small = _font(15)

    # ---------------------------------------------------------------- helpers
    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return self.board_rect.x + x * self.scale, self.board_rect.y + y * self.scale

    def px_to_board(self, px: float, py: float) -> tuple[float, float]:
        return (px - self.board_rect.x) / self.scale, (py - self.board_rect.y) / self.scale

    # ---------------------------------------------------------------- static layer
    def _build_static(self) -> pygame.Surface:
        lv, s = self.level, self.scale
        surf = pygame.Surface((self.width, self.height))
        surf.fill(PANEL_BG)
        surf.blit(wood_texture(self.outer_w, self.outer_h, FRAME_BASE, FRAME_DARK, 34 * s / 2.6, seed=7),
                  (0, 0))
        # bevel on the outer frame
        bevel = max(2, round(1.2 * s))
        pygame.draw.rect(surf, (120, 82, 44), (0, 0, self.outer_w, self.outer_h), bevel)
        board = wood_texture(self.board_w, self.board_h, BOARD_BASE, BOARD_DARK, 26 * s / 2.6, seed=3)

        # guide line
        pts = [(x * s, y * s) for x, y in lv.path.points]
        lw = max(2, round(1.05 * s))
        pygame.draw.lines(board, INK, False, pts, lw)
        for p in pts:
            pygame.draw.circle(board, INK, p, lw / 2)

        # start and finish markers
        fx, fy = lv.finish
        fr = lv.finish_radius * s
        pygame.gfxdraw.filled_circle(board, round(fx * s), round(fy * s), round(fr), (196, 150, 92))
        pygame.gfxdraw.aacircle(board, round(fx * s), round(fy * s), round(fr), INK)
        pygame.gfxdraw.aacircle(board, round(fx * s), round(fy * s), round(fr * 0.6), INK)
        pygame.draw.circle(board, INK, (fx * s, fy * s), max(2, fr * 0.18))
        sx, sy = lv.start
        pygame.gfxdraw.aacircle(board, round(sx * s), round(sy * s), round(lv.ball_radius * 1.5 * s), INK)

        # holes
        R = lv.hole_radius * s
        hole = hole_sprite(R, 1.4 * s)
        for hx, hy in lv.holes:
            board.blit(hole, hole.get_rect(center=(hx * s, hy * s)))

        # printed numbers and texts
        num_font = _font(max(9, round(4.4 * s)), bold=True)
        for i, (hx, hy) in enumerate(lv.holes, 1):
            self._blit_label(board, num_font, str(i), self._label_spot_for_hole(hx, hy, num_font, str(i)))
        txt_font = _font(max(9, round(4.6 * s)), bold=True)
        self._blit_label(board, txt_font, "START", self._label_spot(lv.start, txt_font, "START", 11.0))
        self._blit_label(board, txt_font, "FINISH", self._label_spot(lv.finish, txt_font, "FINISH", 12.5))

        # walls with soft shadows
        shadow = pygame.Surface((self.board_w, self.board_h))
        shadow.fill((255, 255, 255))
        off = (1.8 * s, 2.4 * s)
        for w in lv.walls:
            cs = [(x * s, y * s) for x, y in w.corners]
            moved = [(x + off[0], y + off[1]) for x, y in cs]
            pygame.draw.polygon(shadow, (170, 150, 128), moved)
            for a, b, a2, b2 in zip(cs, cs[1:] + cs[:1], moved, moved[1:] + moved[:1]):
                pygame.draw.polygon(shadow, (170, 150, 128), [a, b, b2, a2])
        # the frame shades the top and left edges of the recessed board
        edge = round(3.0 * s)
        pygame.draw.rect(shadow, (150, 130, 108), (0, 0, self.board_w, edge))
        pygame.draw.rect(shadow, (150, 130, 108), (0, 0, edge, self.board_h))
        shadow = _blur(shadow, max(1, round(1.2 * s)))
        board.blit(shadow, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

        for k, w in enumerate(lv.walls):
            self._draw_wall(board, w, seed=100 + k)
        surf.blit(board, self.board_rect)
        return surf

    def _draw_wall(self, board: pygame.Surface, w, seed: int) -> None:
        s = self.scale
        length, thick = max(1, round(2 * w.hw * s)), max(1, round(2 * w.hh * s))
        strip = wood_texture(length, thick, WALL_BASE, WALL_DARK, 6.5 * s / 2.6, seed)
        rotated = pygame.transform.rotate(strip, -math.degrees(w.angle))
        board.blit(rotated, rotated.get_rect(center=(w.cx * s, w.cy * s)))
        cs = [(x * s, y * s) for x, y in w.corners]
        for a, b in zip(cs, cs[1:] + cs[:1]):
            ex, ey = b[0] - a[0], b[1] - a[1]
            n = math.hypot(ex, ey) or 1.0
            nx, ny = ey / n, -ex / n          # outward normal for clockwise-on-screen corners
            facing = nx * LIGHT[0] + ny * LIGHT[1]
            color = (250, 228, 190) if facing > 0.3 else (128, 88, 48) if facing < -0.3 else (190, 148, 96)
            pygame.draw.line(board, color, a, b, max(1, round(0.5 * s)))

    def _label_ok(self, cx: float, cy: float, w: float, h: float) -> bool:
        lv = self.level
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if x0 < 1 or y0 < 1 or x1 > lv.width - 1 or y1 > lv.height - 1:
            return False
        probes = [(x, y) for x in (x0, cx, x1) for y in (y0, cy, y1)]
        for wall in lv.walls:
            if any(wall.signed_distance(x, y) < 0.8 for x, y in probes):
                return False
        for hx, hy in lv.holes:
            qx, qy = min(max(hx, x0), x1), min(max(hy, y0), y1)
            if math.hypot(hx - qx, hy - qy) < lv.hole_radius + 1.0:
                return False
        for x, y in probes:
            if lv.path.project(x, y)[1] < 1.5:
                return False
        fx, fy = lv.finish
        qx, qy = min(max(fx, x0), x1), min(max(fy, y0), y1)
        return math.hypot(fx - qx, fy - qy) >= lv.finish_radius + 1.0

    def _text_size_mm(self, font: pygame.font.Font, text: str) -> tuple[float, float]:
        w, h = font.size(text)
        return w / self.scale, h * 0.72 / self.scale

    def _label_spot_for_hole(self, hx: float, hy: float, font, text: str) -> tuple[float, float]:
        lv = self.level
        tw, th = self._text_size_mm(font, text)
        _, _, (qx, qy) = lv.path.project(hx, hy)
        base = math.atan2(hy - qy, hx - qx)
        dist = lv.hole_radius + 1.2
        for turn in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
            a = base + math.radians(turn)
            ux, uy = math.cos(a), math.sin(a)
            # push the label out until its box clears the hole
            extra = abs(ux) * tw / 2 + abs(uy) * th / 2
            cx, cy = hx + ux * (dist + extra), hy + uy * (dist + extra)
            if self._label_ok(cx, cy, tw, th):
                return cx, cy
        return hx + dist + tw / 2, hy

    def _label_spot(self, anchor, font, text: str, dist: float) -> tuple[float, float]:
        tw, th = self._text_size_mm(font, text)
        ax, ay = anchor
        for ux, uy in ((0, 1), (0, -1), (1, 0), (-1, 0), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)):
            cx = ax + ux * (dist + abs(ux) * tw / 2)
            cy = ay + uy * (dist + abs(uy) * th / 2)
            if self._label_ok(cx, cy, tw, th):
                return cx, cy
        return ax, ay + dist

    def _blit_label(self, board: pygame.Surface, font, text: str, pos_mm) -> None:
        img = font.render(text, True, INK)
        board.blit(img, img.get_rect(center=(pos_mm[0] * self.scale, pos_mm[1] * self.scale)))

    def _build_tilt_shades(self) -> dict[str, pygame.Surface]:
        """Gradients that darken the low side of the board when it tilts."""
        w, h = self.board_w, self.board_h
        ramp_x = np.linspace(0, 1, w) ** 1.5
        ramp_y = np.linspace(0, 1, h) ** 1.5
        shades = {}
        for key, a in (("right", np.tile(ramp_x, (h, 1))), ("left", np.tile(ramp_x[::-1], (h, 1))),
                       ("down", np.tile(ramp_y[:, None], (1, w))), ("up", np.tile(ramp_y[::-1, None], (1, w)))):
            shades[key] = _rgba_surface(np.zeros((h, w, 3)), a * 255)
        return shades

    # ---------------------------------------------------------------- per frame
    def draw(self, screen: pygame.Surface, game: LabyrinthGame, hud: Hud | None = None,
             debug: bool = False, fall_anim: float | None = None) -> None:
        self._draw_board(screen, game, debug, fall_anim)
        if self.panel_w:
            self._draw_panel(screen, game, hud or Hud())

    def _draw_board(self, screen: pygame.Surface, game: LabyrinthGame, debug: bool,
                    fall_anim: float | None) -> None:
        screen.blit(self.static, (0, 0))
        tx, ty = game.tilt
        for key, amount in (("right", tx), ("left", -tx), ("down", ty), ("up", -ty)):
            if amount > 0.02:
                shade = self.tilt_shades[key]
                shade.set_alpha(round(55 * amount))
                screen.blit(shade, self.board_rect)
        if debug:
            self._draw_sensors(screen, game)
        self._draw_ball(screen, game, fall_anim)

    def _draw_ball(self, screen: pygame.Surface, game: LabyrinthGame, fall_anim: float | None) -> None:
        lv, s = self.level, self.scale
        x, y = game.ball_pos
        depth = game.ball_depth
        fade = 1.0
        if game.status is Status.FELL and game.fell_into is not None:
            hx, hy = lv.holes[game.fell_into]
            t = 1.0 if fall_anim is None else min(1.0, fall_anim)
            k = min(1.0, t * 3)
            x, y = x + (hx - x) * k, y + (hy - y) * k
            depth = 1.0 + t
            fade = max(0.0, 1.0 - t * 1.4)
            if fade <= 0:
                return
        size_k = 1.0 - 0.18 * min(depth, 1.0) - 0.25 * max(0.0, depth - 1.0)
        px, py = self.to_px(x, y)
        if depth < 0.3:
            sh = self.ball_shadow
            screen.blit(sh, sh.get_rect(center=(px + 1.6 * s, py + 2.2 * s)))
        ball = self.ball
        if size_k < 0.999 or fade < 1.0:
            w = max(2, round(ball.get_width() * size_k))
            ball = pygame.transform.smoothscale(ball, (w, w))
            dark = round(255 * max(0.15, 1.0 - 0.45 * depth))
            ball.fill((dark, dark, dark, round(255 * fade)), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(ball, ball.get_rect(center=(px, py)))

    def _draw_sensors(self, screen: pygame.Surface, game: LabyrinthGame) -> None:
        """Visualise what the AI observes: rays, nearby holes and the path ahead."""
        lv, s = self.level, self.scale
        x, y = game.ball_pos
        bx, by = self.to_px(x, y)
        for k, dist in enumerate(game.wall_rays()):
            a = 2 * math.pi * k / 8
            reach = dist + lv.ball_radius
            end = self.to_px(x + math.cos(a) * reach, y + math.sin(a) * reach)
            pygame.draw.line(screen, (70, 140, 230), (bx, by), end, 1)
            pygame.draw.circle(screen, (70, 140, 230), end, 2)
        for dx, dy, _ in game.nearest_holes(4):
            pygame.draw.circle(screen, BAD, self.to_px(x + dx, y + dy), lv.hole_radius * s + 3, 2)
        for px, py in game.path_lookahead((15.0, 40.0, 80.0)):
            pygame.draw.circle(screen, (60, 200, 220), self.to_px(px, py), 4)
        qx, qy = lv.path.point_at(game.progress_s)
        pygame.draw.circle(screen, GOOD, self.to_px(qx, qy), 4)
        vx, vy = game.ball_vel
        pygame.draw.line(screen, ACCENT, (bx, by), (bx + vx * s * 0.15, by + vy * s * 0.15), 3)

    # ---------------------------------------------------------------- side panel
    def _draw_panel(self, screen: pygame.Surface, game: LabyrinthGame, hud: Hud) -> None:
        x0 = self.outer_w + 24
        right = self.width - 24
        y = 22
        title = self.f_title.render(t("LABYRINT"), True, ACCENT)
        screen.blit(title, (x0, y))
        y += title.get_height()
        screen.blit(self.f_small.render(t("Bana: {name}").format(name=t(self.level.name)), True, PANEL_DIM),
                    (x0, y))
        y += 40

        st = game.state
        n_holes = len(self.level.holes)
        rows = [
            (t("Tid"), f"{st.time:5.1f} s"),
            (t("Hål passerade"), f"{st.holes_passed} / {n_holes}"),
            (t("Försök"), str(hud.attempts)),
            (t("Bästa tid"), f"{hud.best_time:.1f} s" if hud.best_time is not None else "–"),
            (t("Rekord"), t("{n} hål").format(n=hud.best_holes)),
        ]
        for label, value in rows:
            screen.blit(self.f_text.render(label, True, PANEL_DIM), (x0, y))
            v = self.f_text.render(value, True, PANEL_TEXT)
            screen.blit(v, (right - v.get_width(), y))
            y += 30

        y += 6
        bar = pygame.Rect(x0, y, right - x0, 12)
        pygame.draw.rect(screen, (62, 52, 44), bar, border_radius=6)
        fill = bar.copy()
        fill.width = round(bar.width * min(1.0, st.max_progress_s / self.level.path.length))
        if fill.width > 0:
            pygame.draw.rect(screen, ACCENT, fill, border_radius=6)
        y += 34

        # the two knobs, turned according to the current tilt
        for i, (label, value) in enumerate(((t("X-ratt"), st.tilt[0]), (t("Y-ratt"), st.tilt[1]))):
            cx = x0 + 52 + i * 140
            cy = y + 38
            self._draw_knob(screen, cx, cy, value)
            img = self.f_small.render(label, True, PANEL_DIM)
            screen.blit(img, img.get_rect(center=(cx, cy + 50)))
        y += 112

        controller = t("Styrs av: {controller}").format(controller=t(hud.controller))
        screen.blit(self.f_text.render(controller, True, PANEL_TEXT), (x0, y))
        y += 32
        for line in hud.info.splitlines():
            screen.blit(self.f_small.render(line, True, PANEL_DIM), (x0, y))
            y += 21
        y += 8

        message = t("PAUS – tryck P") if hud.paused else hud.message
        color = ACCENT if hud.paused else hud.message_color
        for line in self._wrap(message, self.f_big, right - x0):
            img = self.f_big.render(line, True, color)
            screen.blit(img, (x0, y))
            y += img.get_height()

        help_lines = [
            t("Mus: luta brädet"),
            t("Pilar/WASD: vrid rattarna"),
            t("Mellanslag: plant bräde"),
            t("R: börja om    P: paus"),
            t("F1: visa AI-sensorer"),
            t("F2: växla autopilot / AI"),
            t("Esc: avsluta"),
        ]
        y = self.height - 22 - len(help_lines) * 21
        for line in help_lines:
            screen.blit(self.f_small.render(line, True, PANEL_DIM), (x0, y))
            y += 21

    def _draw_knob(self, screen: pygame.Surface, cx: int, cy: int, value: float) -> None:
        r = 30
        pygame.gfxdraw.filled_circle(screen, cx + 3, cy + 4, r, (20, 15, 12))
        for k in range(r, 0, -1):
            shade = round(18 + 30 * (1 - k / r))
            pygame.gfxdraw.filled_circle(screen, cx - (r - k) // 4, cy - (r - k) // 4, k, (shade, shade, shade + 2))
        pygame.gfxdraw.aacircle(screen, cx, cy, r, (70, 70, 74))
        a = math.radians(-90 + value * 135)
        end = (cx + math.cos(a) * (r - 6), cy + math.sin(a) * (r - 6))
        pygame.draw.line(screen, (235, 235, 235), (cx, cy), end, 3)

    @staticmethod
    def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
        lines = []
        for para in text.split("\n"):
            line = ""
            for word in para.split():
                trial = f"{line} {word}".strip()
                if font.size(trial)[0] <= width or not line:
                    line = trial
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines

    # ---------------------------------------------------------------- pixels for AI / video
    def board_array(self, game: LabyrinthGame, debug: bool = False) -> np.ndarray:
        """The board (with frame, without panel) as an (H, W, 3) uint8 array."""
        surf = pygame.Surface((self.outer_w, self.outer_h))
        self._draw_board(surf, game, debug, None)
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2).copy()
