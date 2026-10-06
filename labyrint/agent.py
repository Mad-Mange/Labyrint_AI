"""A trained neural network as a controller for LabyrinthGame (same interface as PathFollower).

    python -m labyrint.agent models/labyrint_ai.zip --episodes 20

Needs torch and stable-baselines3 (imported lazily, so the game itself runs without them).

LivePilot follows a training run instead: train.py writes a policy snapshot every few
seconds (write_live_snapshot) and `python play.py --live` plays with the newest one.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from .env import observe
from .game import LabyrinthGame, Status

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "labyrint_ai.zip"
RUNS = Path(__file__).resolve().parents[1] / "runs"
LIVE_POLICY = "live_policy.pt"   # in the run folder, replaced every few seconds during training
LIVE_INFO = "live.json"          # {"steps", "total", "policy_class"}, written after the policy


def latest_run(runs: Path = RUNS) -> Path | None:
    """The most recently started training run (folder with args.json), or None."""
    started = list(runs.glob("*/args.json"))
    return max(started, key=lambda p: p.stat().st_mtime).parent if started else None


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


def write_live_snapshot(policy, run_dir: Path, info: dict) -> bool:
    """Save a training policy for LivePilot. Returns False if a viewer is reading it right now."""
    cls = type(policy)
    meta = {**info, "policy_class": f"{cls.__module__}:{cls.__qualname__}"}
    try:
        # Write to a temporary file and rename, so a reader never sees half a file.
        policy.save(str(run_dir / (LIVE_POLICY + ".tmp")))
        os.replace(run_dir / (LIVE_POLICY + ".tmp"), run_dir / LIVE_POLICY)
        (run_dir / (LIVE_INFO + ".tmp")).write_text(json.dumps(meta))
        os.replace(run_dir / (LIVE_INFO + ".tmp"), run_dir / LIVE_INFO)
    except PermissionError:  # Windows refuses to replace a file another process has open
        return False
    return True


class LivePilot(NeuralPilot):
    """A NeuralPilot that follows a training run: reload() swaps in the newest snapshot."""

    def __init__(self, run_dir: str | Path, frame_skip: int = 2):
        super().__init__(None, frame_skip)
        self.run_dir = Path(run_dir)
        self.info: dict | None = None   # live.json of the loaded snapshot
        self._stamp: int | None = None

    def reload(self) -> bool:
        """Load the newest snapshot if the training has written a new one since last time."""
        try:
            stamp = (self.run_dir / LIVE_INFO).stat().st_mtime_ns
        except FileNotFoundError:
            return False
        if stamp == self._stamp:
            return False
        try:
            info = json.loads((self.run_dir / LIVE_INFO).read_text())
            module, name = info["policy_class"].split(":")
            cls = getattr(importlib.import_module(module), name)
            policy = cls.load(str(self.run_dir / LIVE_POLICY), device="cpu")
        except (OSError, ValueError, KeyError, EOFError, RuntimeError, pickle.UnpicklingError):
            return False  # caught it while the training was replacing the files - try again later
        self.model, self.info, self._stamp = policy, info, stamp
        self._tick = None
        return True


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
