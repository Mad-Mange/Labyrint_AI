import numpy as np
import pytest

from labyrint.env import LabyrinthEnv, observe

sb3 = pytest.importorskip("stable_baselines3")


def test_observe_matches_env():
    env = LabyrinthEnv(random_start=1.0)
    obs, _ = env.reset(seed=4)
    assert np.array_equal(obs, observe(env.game))


def test_neural_pilot_replays_env_trajectory(tmp_path):
    """A model saved by training must drive a plain game exactly like it drove the env."""
    from labyrint.agent import NeuralPilot

    env = LabyrinthEnv(random_start=1.0)
    model = sb3.PPO("MlpPolicy", env, seed=0, device="cpu",
                    policy_kwargs=dict(log_std_init=-0.5))
    model.save(tmp_path / "m")
    pilot = NeuralPilot(tmp_path / "m.zip", frame_skip=env.frame_skip)

    obs, _ = env.reset(seed=7)
    game = env.game.clone()
    for _ in range(150):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        for _ in range(env.frame_skip):
            game.step(pilot.act(game))
        assert game.state == env.game.state
        if terminated or truncated:
            break
