"""Träna en AI att spela Labyrint (förstärkningsinlärning med stable-baselines3).

    python train.py                              # PPO, 64 parallella miljöer, 30 M steg
    python train.py --steps 5e6 --name test      # kort körning
    python train.py --resume runs/ppo/latest_model.zip
    tensorboard --logdir runs                    # följ träningen i webbläsaren

Allt hamnar i runs/<namn>/: best_model.zip (bäst på riktiga omgångar från START),
latest_model.zip, vecnormalize.pkl och tensorboard-loggar.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from labyrint.env import LabyrinthEnv
from labyrint.vecenv import BatchedSubprocVecEnv

ROOT = Path(__file__).resolve().parent
AUTOPILOT_SECONDS = 38.0   # the hand-written baseline on "classic"


def make_env(level: str, random_start: float, max_seconds: float):
    def _init():
        env = LabyrinthEnv(level, random_start=random_start, max_episode_seconds=max_seconds)
        return Monitor(env, info_keywords=("is_success", "holes_passed"))
    return _init


def evaluate(model, level: str, episodes: int, seed: int = 10_000) -> dict:
    """Play `episodes` full games from START in lockstep (one batched forward pass per step)."""
    envs = [LabyrinthEnv(level) for _ in range(episodes)]
    obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    active = list(range(episodes))
    results: list[dict | None] = [None] * episodes
    while active:
        actions, _ = model.predict(obs[active], deterministic=True)
        still = []
        for i, a in zip(active, actions):
            obs[i], _, terminated, truncated, info = envs[i].step(a)
            if terminated or truncated:
                results[i] = info
            else:
                still.append(i)
        active = still
    wins = [r for r in results if r["is_success"]]
    return {
        "success_rate": len(wins) / episodes,
        "mean_time": float(np.mean([r["time"] for r in wins])) if wins else float("nan"),
        "best_time": min((r["time"] for r in wins), default=float("nan")),
        "holes_passed": float(np.mean([r["holes_passed"] for r in results])),
    }


class EvalCallback(BaseCallback):
    """Every `freq` steps: play whole games from START, log them and keep the best model."""

    def __init__(self, out: Path, level: str, freq: int, episodes: int):
        super().__init__()
        self.out, self.level, self.freq, self.episodes = out, level, freq, episodes
        self.best_key: tuple[float, float] | None = None
        self.next_eval = freq
        self.last_eval = 0
        self.t0 = time.perf_counter()
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.freq
            self.run_eval()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps - self.last_eval > self.freq // 4:
            self.run_eval()

    def run_eval(self) -> None:
        self.last_eval = self.num_timesteps
        r = evaluate(self.model, self.level, self.episodes)
        for k, v in r.items():
            if not np.isnan(v):
                self.logger.record(f"eval/{k}", v)
        # Best = most finishes, then fastest; before the first finish: most holes passed.
        key = (r["success_rate"], -r["mean_time"] if r["success_rate"] > 0 else r["holes_passed"] - 1e3)
        best = self.best_key is None or key > self.best_key
        self.model.save(self.out / "latest_model")
        self._save_vecnormalize()
        if best:
            self.best_key = key
            self.model.save(self.out / "best_model")
        elapsed = time.perf_counter() - self.t0
        self.history.append({"steps": self.num_timesteps, "minutes": elapsed / 60, **r})
        (self.out / "eval_history.json").write_text(json.dumps(self.history, indent=1))
        tid = f"{r['mean_time']:5.1f} s (bäst {r['best_time']:.1f})" if r["success_rate"] else "   –   "
        print(f"[{elapsed / 60:5.1f} min] {self.num_timesteps / 1e6:6.2f} M steg | i mål "
              f"{r['success_rate']:4.0%} | tid {tid} | hål passerade {r['holes_passed']:4.1f}"
              f"{'  <- ny bästa' if best else ''}", flush=True)

    def _save_vecnormalize(self) -> None:
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(self.out / "vecnormalize.pkl"))


def build_model(algo: str, venv, args, tb_log: str):
    if algo == "ppo":
        return PPO(
            "MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs,
            gamma=args.gamma, gae_lambda=0.95, learning_rate=args.lr, clip_range=0.2,
            ent_coef=0.0, max_grad_norm=0.5, vf_coef=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
            tensorboard_log=tb_log, device=args.device, seed=args.seed, verbose=0)
    return SAC(
        "MlpPolicy", venv, learning_rate=args.lr, buffer_size=1_000_000, batch_size=512,
        gamma=args.gamma, train_freq=1, gradient_steps=max(1, args.envs // 2), learning_starts=10_000,
        policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log=tb_log, device=args.device,
        seed=args.seed, verbose=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Träna en AI att spela Labyrint.")
    ap.add_argument("--algo", choices=("ppo", "sac"), default="ppo")
    ap.add_argument("--level", default="classic")
    ap.add_argument("--steps", type=float, default=30e6, help="antal miljösteg (1 steg = 1/30 s spel)")
    ap.add_argument("--envs", type=int, default=64, help="parallella miljöer")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processer som delar på miljöerna")
    ap.add_argument("--name", default=None, help="namn på körningen (mapp under runs/)")
    ap.add_argument("--random-start", type=float, default=0.5,
                    help="andel omgångar som startar på en slumpvis punkt längs linjen")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="max speltid per omgång")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=256, help="PPO: steg per miljö mellan uppdateringar")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=10, help="PPO: genomgångar av varje omgång data")
    ap.add_argument("--eval-freq", type=float, default=1e6, help="utvärdera var N:e steg")
    ap.add_argument("--eval-episodes", type=int, default=20)
    # PPO with a small MLP is as fast on the CPU as on the GPU here (env stepping dominates),
    # and the CPU avoids sporadic CUDA driver errors on a GPU that also drives the display.
    ap.add_argument("--device", default="cpu", help="cpu, cuda eller auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="fortsätt träna från en sparad modell (.zip)")
    args = ap.parse_args()

    name = args.name or args.algo
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    fns = [make_env(args.level, args.random_start, args.max_seconds)] * args.envs
    venv = BatchedSubprocVecEnv(fns, args.workers) if args.workers > 1 else DummyVecEnv(fns)
    # Rewards are rescaled during training only; observations are already in [-1, 1],
    # so a trained model needs no normalisation statistics to play.
    resume = Path(args.resume) if args.resume else None
    vn_path = resume.with_name("vecnormalize.pkl") if resume else None
    if vn_path is not None and vn_path.exists():
        venv = VecNormalize.load(str(vn_path), venv)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=args.gamma)

    if resume:
        cls = PPO if args.algo == "ppo" else SAC
        model = cls.load(resume, env=venv, device=args.device, tensorboard_log=str(ROOT / "runs"))
    else:
        model = build_model(args.algo, venv, args, str(ROOT / "runs"))

    print(f"Tränar {args.algo.upper()} på '{args.level}' med {args.envs} miljöer i {args.steps / 1e6:g} M steg "
          f"({model.device}). Autopiloten klarar banan på ca {AUTOPILOT_SECONDS:.0f} s.", flush=True)
    callback = EvalCallback(out, args.level, int(args.eval_freq), args.eval_episodes)
    try:
        model.learn(total_timesteps=int(args.steps), callback=callback, tb_log_name=name,
                    reset_num_timesteps=resume is None)
    except KeyboardInterrupt:
        print("Avbruten – sparar.")
        model.save(out / "latest_model")
        callback._save_vecnormalize()
    venv.close()
    print(f"\nKlart. Bästa modellen: {out / 'best_model.zip'}\n"
          f"Titta på den:  python play.py --ai {(out / 'best_model.zip').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
