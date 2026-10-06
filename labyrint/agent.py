"""A trained neural network as a controller for LabyrinthGame (same interface as PathFollower).

    python -m labyrint.agent models/labyrint_ai.zip --episodes 20

Needs torch and stable-baselines3 (imported lazily, so the game itself runs without them).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .env import observe
from .game import LabyrinthGame, Status

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "labyrint_ai.zip"


def load_model(path: str | Path, device: str = "cpu"):
    """Load a stable-baselines3 model without knowing which algorithm trained it."""
    from stable_baselines3 import PPO, SAC

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    errors = []
    for algo in (PPO, SAC):
        try:
            return algo.load(path, device=device)
        except Exception as e:  # wrong algorithm for this file - try the next one
            errors.append(f"{algo.__name__}: {e}")
    raise ValueError(f"could not load {path}: " + "; ".join(errors))


class NeuralPilot:
    """Drives a LabyrinthGame with a trained policy.

    The policy was trained to decide every `frame_skip` ticks and hold its action in
    between, so it is called on the same ticks here as in the env.
    """

    def __init__(self, model, frame_skip: int = 2, deterministic: bool = True):
        self.model = load_model(model) if isinstance(model, (str, Path)) else model
        self.frame_skip = frame_skip
        self.deterministic = deterministic
        self._tick: int | None = None
        self._action = (0.0, 0.0)

    def act(self, game: LabyrinthGame) -> tuple[float, float]:
        tick = round(game.time * game.physics.tick_rate)
        if tick % self.frame_skip == 0 or self._tick is None or tick != self._tick + 1:
            a, _ = self.model.predict(observe(game), deterministic=self.deterministic)
            a = np.clip(a, -1.0, 1.0)
            self._action = (float(a[0]), float(a[1]))
        self._tick = tick
        return self._action


def main() -> None:
    from .autopilot import run_episode

    ap = argparse.ArgumentParser(description="Run a trained model headless and measure it.")
    ap.add_argument("model", nargs="?", default=str(DEFAULT_MODEL))
    ap.add_argument("--level", default="classic")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=2.0, help="random start offset (mm)")
    args = ap.parse_args()

    game = LabyrinthGame(args.level, seed=0)
    pilot = NeuralPilot(args.model)
    wins, times = 0, []
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        status, t, holes = run_episode(game, pilot, jitter=args.jitter)
        wins += status is Status.FINISHED
        if status is Status.FINISHED:
            times.append(t)
        extra = "" if status is Status.FINISHED else f" (hole {game.fell_into + 1 if game.fell_into is not None else '-'})"
        print(f"episode {ep + 1:3d}: {status.value:8s} {t:6.1f} s  passed {holes:2d} holes{extra}")
    best = f", best {min(times):.1f} s, mean {sum(times) / len(times):.1f} s" if times else ""
    print(f"\n{wins}/{args.episodes} finished{best} ({time.perf_counter() - t0:.1f} s wall time)")


if __name__ == "__main__":
    main()
