import json

import numpy as np
import pytest

from labyrint.env import LabyrinthEnv

sb3 = pytest.importorskip("stable_baselines3")


def test_live_pilot_follows_snapshots(tmp_path):
    from labyrint.agent import LivePilot, write_live_snapshot

    env = LabyrinthEnv()
    model = sb3.PPO("MlpPolicy", env, seed=0, device="cpu")
    pilot = LivePilot(tmp_path, frame_skip=env.frame_skip)
    assert not pilot.reload() and pilot.model is None     # the training has not started yet

    assert write_live_snapshot(model.policy, tmp_path, {"steps": 123, "total": 1000})
    assert pilot.reload() and pilot.info["steps"] == 123
    assert not pilot.reload()                              # nothing new since last time

    obs, _ = env.reset(seed=1)
    expected, _ = model.predict(obs, deterministic=True)
    assert np.allclose(pilot.act(env.game), np.clip(expected, -1.0, 1.0))


def test_training_writes_live_files(tmp_path):
    """train.py's LiveCallback feeds dashboard.py (progress.jsonl) and play.py --live (snapshot)."""
    pytest.importorskip("matplotlib")
    from dashboard import ProgressLog
    from train import LiveCallback

    model = sb3.PPO("MlpPolicy", LabyrinthEnv(), n_steps=64, batch_size=64, n_epochs=1, seed=0, device="cpu")
    model.learn(total_timesteps=192, callback=LiveCallback(tmp_path, snapshot_seconds=0.0))

    log = ProgressLog(tmp_path / "progress.jsonl")
    assert log.poll()
    assert [r["step"] for r in log.rows] == [64, 128, 192]
    assert log.last("time/target_timesteps") == 192
    xs, ys = log.series("train/value_loss")
    assert len(xs) == 2 and all(y >= 0 for y in ys)      # losses exist after the first update
    assert json.loads((tmp_path / "live.json").read_text())["steps"] == 192
    assert not log.poll()                                  # nothing new


def test_run_level_reads_args_json(tmp_path):
    from labyrint.agent import run_level

    assert run_level(tmp_path) is None and run_level(tmp_path / "best_model.zip") is None
    (tmp_path / "args.json").write_text(json.dumps({"level": "pinnar", "steps": 1e6}))
    assert run_level(tmp_path) == "pinnar"
    assert run_level(tmp_path / "best_model.zip") == "pinnar"   # a model saved in the run folder
