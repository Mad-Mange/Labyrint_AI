import numpy as np
import pytest

from labyrint.env import LabyrinthEnv

pytest.importorskip("stable_baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from labyrint.vecenv import BatchedSubprocVecEnv  # noqa: E402


def make():
    return LabyrinthEnv(random_start=1.0, max_episode_seconds=3.0)


def test_batched_subproc_matches_dummy():
    """Several envs per worker process must behave exactly like SB3's in-process DummyVecEnv."""
    n = 5
    ref, venv = DummyVecEnv([make] * n), BatchedSubprocVecEnv([make] * n, n_workers=2)
    try:
        for v in (ref, venv):
            v.seed(3)
        assert np.array_equal(ref.reset(), venv.reset())
        rng = np.random.default_rng(0)
        for _ in range(200):   # long enough for episodes to end and auto-reset
            actions = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
            o1, r1, d1, i1 = ref.step(actions)
            o2, r2, d2, i2 = venv.step(actions)
            assert np.array_equal(o1, o2) and np.allclose(r1, r2) and np.array_equal(d1, d2)
            for a, b in zip(i1, i2):
                assert a["TimeLimit.truncated"] == b["TimeLimit.truncated"]
                if "terminal_observation" in a:
                    assert np.array_equal(a["terminal_observation"], b["terminal_observation"])
        assert venv.get_attr("frame_skip", indices=[4, 0]) == [2, 2]
        assert venv.env_method("get_wrapper_attr", "random_start", indices=[3]) == [1.0]
    finally:
        venv.close()
