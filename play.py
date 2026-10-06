"""Spela Labyrint!

    python play.py                 # spela själv med mus eller tangentbord
    python play.py --a             # titta på den inbyggda autopiloten
    python play.py --ai            # titta på den tränade AI:n (models/labyrint_ai.zip)
    python play.py --ai runs/ppo/best_model.zip
    python play.py --live          # titta på AI:n medan den tränar (senaste körningen i runs/)
    python play.py --live runs/live-20261006-1830
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pygame

from labyrint import LabyrinthGame, Status, available_levels
from labyrint.agent import DEFAULT_MODEL, LivePilot, NeuralPilot, latest_run
from labyrint.autopilot import PathFollower
from labyrint.env import STALL_MM, STALL_SECONDS
from labyrint.geometry import clamp
from labyrint.render import BAD, GOOD, PANEL_TEXT, Hud, Renderer

KNOB_SPEED = 1.6          # tilt units per second while an arrow key is held
FALL_ANIM_SECONDS = 1.3
WELCOME = "Luta brädet och för kulan från START till FINISH!"
LIVE = "AI (live)"        # controller name for a model that is still training
LIVE_PAUSE = 2.0          # seconds to show a finish/stall before the next live attempt


def main() -> None:
    ap = argparse.ArgumentParser(description="Labyrint – det klassiska kulspelet.")
    ap.add_argument("--level", default="classic", choices=available_levels())
    ap.add_argument("--scale", type=float, default=2.6, help="pixlar per millimeter")
    ap.add_argument("--mouse-range", type=float, default=0.45,
                    help="hur långt från mitten (andel av halva brädet) musen ger full lutning")
    ap.add_argument("--a", "-a", "--autopilot", dest="autopilot", action="store_true", help="låt autopiloten spela")
    ap.add_argument("--ai", nargs="?", const=str(DEFAULT_MODEL), default=None, metavar="MODELL",
                    help="låt en tränad AI spela (standard: models/labyrint_ai.zip)")
    ap.add_argument("--live", nargs="?", const="", default=None, metavar="KÖRNING",
                    help="titta på AI:n medan den tränar (standard: senaste körningen i runs/)")
    args = ap.parse_args()

    live: LivePilot | None = None
    if args.live is not None:
        run = Path(args.live) if args.live else latest_run()
        if run is None:
            raise SystemExit("Hittar ingen träning i runs/. Starta en med: python train.py")
        live = LivePilot(run)
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "10,40")  # leaves room for dashboard.py

    pygame.init()
    game = LabyrinthGame(args.level)
    renderer = Renderer(game.level, args.scale)
    screen = pygame.display.set_mode((renderer.width, renderer.height))
    pygame.display.set_caption("Labyrint – AI:n tränar" if live else "Labyrint")
    clock = pygame.time.Clock()
    pilot = PathFollower()
    ai_path = Path(args.ai) if args.ai else DEFAULT_MODEL
    ai: NeuralPilot | None = None

    hud = Hud(controller="Autopilot" if args.autopilot else "Mus", message=WELCOME)
    if live:
        hud.controller, hud.message = LIVE, ""

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

    if args.ai and not args.autopilot and not live and load_ai():
        hud.controller = "AI"
    target = [0.0, 0.0]
    tick = 1.0 / game.physics.tick_rate
    accumulator = 0.0
    fall_t: float | None = None
    finish_recorded = False
    debug = False
    # Live mode only: end an attempt when the ball stops making progress, like the training does.
    stall_s, stall_t, stalled, end_t = 0.0, 0.0, False, 0.0

    def live_info() -> str:
        if live.info is None:
            return f"Väntar på träningen\n{live.run_dir.name} ..."
        steps, total = live.info["steps"], live.info["total"]
        minutes = steps * live.frame_skip / game.physics.tick_rate / 60
        practice = f"{minutes / 60:.1f} timmars" if minutes >= 90 else f"{minutes:.0f} minuters"
        return f"Tränad {steps / 1e6:.1f} av {total / 1e6:.0f} M steg\n= ca {practice} övning"

    def restart(count_attempt: bool) -> None:
        nonlocal fall_t, finish_recorded, accumulator, stall_s, stall_t, stalled, end_t
        game.reset()
        if count_attempt:
            hud.attempts += 1
        hud.message, hud.message_color = ("" if hud.controller == LIVE else WELCOME), PANEL_TEXT
        fall_t, finish_recorded, accumulator = None, False, 0.0
        stall_s, stall_t, stalled, end_t = game.max_progress_s, 0.0, False, 0.0
        target[:] = [0.0, 0.0]
        if hud.controller == LIVE:
            live.reload()   # every attempt is played with the newest snapshot

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
                    order += [LIVE] if live else []
                    current = hud.controller if hud.controller in order else "Mus"
                    hud.controller = order[(order.index(current) + 1) % len(order)]
                    if hud.controller == "AI" and not load_ai():
                        hud.controller = "Mus"
                    stalled = False
                    if hud.controller == LIVE:
                        restart(count_attempt=False)
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

        live_mode = hud.controller == LIVE
        waiting = live_mode and live.model is None
        if waiting:
            live.reload()   # the training has not written its first snapshot yet

        if not hud.paused:
            if game.status is Status.RUNNING and not stalled and not waiting:
                accumulator += dt
                while accumulator >= tick and game.status is Status.RUNNING:
                    if hud.controller == "Autopilot":
                        target[:] = pilot.act(game)
                    elif hud.controller == "AI":
                        target[:] = ai.act(game)
                    elif live_mode:
                        target[:] = live.act(game)
                    game.step(target)
                    accumulator -= tick
                hud.best_holes = max(hud.best_holes, game.state.holes_passed)
                if game.max_progress_s >= stall_s + STALL_MM:
                    stall_s, stall_t = game.max_progress_s, game.time
                elif live_mode and game.time - stall_t >= STALL_SECONDS:
                    stalled = True
                    hud.message, hud.message_color = "Kulan står still.\nNytt försök!", BAD

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
                hud.message = f"I MÅL på {game.time:.1f} s!" + ("" if live_mode else "\nTryck R för att spela igen.")
                hud.message_color = GOOD
            if live_mode and (stalled or game.status is Status.FINISHED):
                end_t += dt
                if end_t >= LIVE_PAUSE:
                    restart(count_attempt=True)

        hud.info = live_info() if hud.controller == LIVE else ""

        anim = fall_t / FALL_ANIM_SECONDS if fall_t is not None else None
        renderer.draw(screen, game, hud, debug=debug, fall_anim=anim)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
