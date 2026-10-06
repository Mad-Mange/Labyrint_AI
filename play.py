"""Spela Labyrint!

    python play.py                 # spela själv med mus eller tangentbord
    python play.py --a             # titta på den inbyggda autopiloten
    python play.py --ai            # titta på den tränade AI:n (models/labyrint_ai.zip)
    python play.py --ai runs/ppo/best_model.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from labyrint import LabyrinthGame, Status, available_levels
from labyrint.agent import DEFAULT_MODEL, NeuralPilot
from labyrint.autopilot import PathFollower
from labyrint.geometry import clamp
from labyrint.render import BAD, GOOD, PANEL_TEXT, Hud, Renderer

KNOB_SPEED = 1.6          # tilt units per second while an arrow key is held
FALL_ANIM_SECONDS = 1.3
WELCOME = "Luta brädet och för kulan från START till FINISH!"


def main() -> None:
    ap = argparse.ArgumentParser(description="Labyrint – det klassiska kulspelet.")
    ap.add_argument("--level", default="classic", choices=available_levels())
    ap.add_argument("--scale", type=float, default=2.6, help="pixlar per millimeter")
    ap.add_argument("--mouse-range", type=float, default=0.45,
                    help="hur långt från mitten (andel av halva brädet) musen ger full lutning")
    ap.add_argument("--a", "-a", "--autopilot", dest="autopilot", action="store_true", help="låt autopiloten spela")
    ap.add_argument("--ai", nargs="?", const=str(DEFAULT_MODEL), default=None, metavar="MODELL",
                    help="låt en tränad AI spela (standard: models/labyrint_ai.zip)")
    args = ap.parse_args()

    pygame.init()
    game = LabyrinthGame(args.level)
    renderer = Renderer(game.level, args.scale)
    screen = pygame.display.set_mode((renderer.width, renderer.height))
    pygame.display.set_caption("Labyrint")
    clock = pygame.time.Clock()
    pilot = PathFollower()
    ai_path = Path(args.ai) if args.ai else DEFAULT_MODEL
    ai: NeuralPilot | None = None

    hud = Hud(controller="Autopilot" if args.autopilot else "Mus", message=WELCOME)

    def load_ai() -> bool:
        """Load the trained model the first time it is needed (torch takes a moment to start)."""
        nonlocal ai
        if ai is None:
            try:
                ai = NeuralPilot(ai_path)
            except (ImportError, OSError, ValueError) as e:
                hud.message, hud.message_color = f"Kunde inte ladda AI:n: {e}", BAD
                return False
        return True

    if args.ai and not args.autopilot and load_ai():
        hud.controller = "AI"
    target = [0.0, 0.0]
    tick = 1.0 / game.physics.tick_rate
    accumulator = 0.0
    fall_t: float | None = None
    finish_recorded = False
    debug = False

    def restart(count_attempt: bool) -> None:
        nonlocal fall_t, finish_recorded, accumulator
        game.reset()
        if count_attempt:
            hud.attempts += 1
        hud.message, hud.message_color = WELCOME, PANEL_TEXT
        fall_t, finish_recorded, accumulator = None, False, 0.0
        target[:] = [0.0, 0.0]

    running = True
    while running:
        dt = min(clock.tick(game.physics.tick_rate) / 1000.0, 0.1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    restart(count_attempt=game.time > 0.5 and game.status is not Status.FINISHED)
                elif event.key == pygame.K_p:
                    hud.paused = not hud.paused
                elif event.key == pygame.K_F1:
                    debug = not debug
                elif event.key == pygame.K_F2:
                    order = ["Mus", "Autopilot"] + (["AI"] if ai is not None or ai_path.exists() else [])
                    current = hud.controller if hud.controller in order else "Mus"
                    hud.controller = order[(order.index(current) + 1) % len(order)]
                    if hud.controller == "AI" and not load_ai():
                        hud.controller = "Mus"
                elif event.key == pygame.K_SPACE and hud.controller not in ("Autopilot", "AI"):
                    hud.controller = "Tangentbord"
                    target[:] = [0.0, 0.0]
            elif event.type == pygame.MOUSEMOTION and hud.controller == "Tangentbord":
                if abs(event.rel[0]) + abs(event.rel[1]) > 3:
                    hud.controller = "Mus"

        keys = pygame.key.get_pressed()
        kx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        ky = (keys[pygame.K_DOWN] or keys[pygame.K_s]) - (keys[pygame.K_UP] or keys[pygame.K_w])
        if (kx or ky) and hud.controller == "Mus":
            hud.controller = "Tangentbord"
        if hud.controller == "Tangentbord":
            target[0] = clamp(target[0] + kx * KNOB_SPEED * dt, -1.0, 1.0)
            target[1] = clamp(target[1] + ky * KNOB_SPEED * dt, -1.0, 1.0)
        elif hud.controller == "Mus":
            mx, my = pygame.mouse.get_pos()
            span_x = renderer.board_w / 2 * args.mouse_range
            span_y = renderer.board_h / 2 * args.mouse_range
            target[0] = clamp((mx - renderer.board_rect.centerx) / span_x, -1.0, 1.0)
            target[1] = clamp((my - renderer.board_rect.centery) / span_y, -1.0, 1.0)

        if not hud.paused:
            if game.status is Status.RUNNING:
                accumulator += dt
                while accumulator >= tick and game.status is Status.RUNNING:
                    if hud.controller == "Autopilot":
                        target[:] = pilot.act(game)
                    elif hud.controller == "AI":
                        target[:] = ai.act(game)
                    game.step(target)
                    accumulator -= tick
                hud.best_holes = max(hud.best_holes, game.state.holes_passed)

            if game.status is Status.FELL:
                if fall_t is None:
                    fall_t = 0.0
                    hud.message = f"Kulan föll i hål {game.fell_into + 1}!"
                    hud.message_color = BAD
                fall_t += dt
                if fall_t >= FALL_ANIM_SECONDS:
                    restart(count_attempt=True)
            elif game.status is Status.FINISHED and not finish_recorded:
                finish_recorded = True
                if hud.best_time is None or game.time < hud.best_time:
                    hud.best_time = game.time
                hud.message = f"I MÅL på {game.time:.1f} s!\nTryck R för att spela igen."
                hud.message_color = GOOD

        anim = fall_t / FALL_ANIM_SECONDS if fall_t is not None else None
        renderer.draw(screen, game, hud, debug=debug, fall_anim=anim)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
