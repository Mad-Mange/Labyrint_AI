"""A stable-baselines3 VecEnv that runs several environments in each worker process.

SB3's SubprocVecEnv gives every env its own process and pays one pipe round trip per
env and step - more than a Labyrint step itself costs. Here each worker steps a whole
batch of envs and answers with a single message, which is several times faster.
Behaviour (auto-reset, `terminal_observation`, `TimeLimit.truncated`) matches SubprocVecEnv.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvIndices


def _worker(remote, parent_remote, fns: CloudpickleWrapper) -> None:
    parent_remote.close()
    envs = [fn() for fn in fns.var]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for env, action in zip(envs, data):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    reset_info = {}
                    if done:
                        info["terminal_observation"] = obs
                        obs, reset_info = env.reset()
                    out.append((obs, reward, done, info, reset_info))
                remote.send(out)
            elif cmd == "reset":
                remote.send([env.reset(seed=seed, **({"options": opts} if opts else {}))
                             for env, (seed, opts) in zip(envs, data)])
            elif cmd == "get_attr":
                name, idx = data
                remote.send([envs[i].get_wrapper_attr(name) for i in idx])
            elif cmd == "has_attr":
                name, idx = data
                remote.send([_has_wrapper_attr(envs[i], name) for i in idx])
            elif cmd == "set_attr":
                name, value, idx = data
                remote.send([setattr(envs[i], name, value) for i in idx])
            elif cmd == "env_method":
                name, args, kwargs, idx = data
                remote.send([envs[i].get_wrapper_attr(name)(*args, **kwargs) for i in idx])
            elif cmd == "is_wrapped":
                wrapper, idx = data
                remote.send([is_wrapped(envs[i], wrapper) for i in idx])
            elif cmd == "render":
                remote.send([env.render() for env in envs])
            elif cmd == "get_spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "close":
                for env in envs:
                    env.close()
                break
            else:
                raise NotImplementedError(cmd)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        remote.close()


def _has_wrapper_attr(env: gym.Env, name: str) -> bool:
    try:
        env.get_wrapper_attr(name)
        return True
    except AttributeError:
        return False


class BatchedSubprocVecEnv(VecEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], n_workers: int | None = None,
                 start_method: str | None = None):
        n = len(env_fns)
        n_workers = max(1, min(n_workers or os.cpu_count() or 1, n))
        self.chunks = [list(map(int, c)) for c in np.array_split(np.arange(n), n_workers)]
        self.where = [(w, j) for w, chunk in enumerate(self.chunks) for j in range(len(chunk))]
        ctx = mp.get_context(start_method or ("forkserver" if "forkserver" in mp.get_all_start_methods()
                                              else "spawn"))
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(n_workers)])
        self.processes = []
        for work_remote, remote, chunk in zip(work_remotes, self.remotes, self.chunks):
            fns = CloudpickleWrapper([env_fns[i] for i in chunk])
            p = ctx.Process(target=_worker, args=(work_remote, remote, fns), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()
        self.waiting = False
        self.closed = False
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(n, observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, chunk in zip(self.remotes, self.chunks):
            remote.send(("step", actions[chunk]))
        self.waiting = True

    def step_wait(self):
        results = [r for remote in self.remotes for r in remote.recv()]
        self.waiting = False
        obs, rewards, dones, infos, self.reset_infos = zip(*results)
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones), list(infos)

    def reset(self):
        for remote, chunk in zip(self.remotes, self.chunks):
            remote.send(("reset", [(self._seeds[i], self._options[i]) for i in chunk]))
        results = [r for remote in self.remotes for r in remote.recv()]
        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return np.stack(obs)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True

    def get_images(self):
        for remote in self.remotes:
            remote.send(("render", None))
        return [img for remote in self.remotes for img in remote.recv()]

    def _call(self, cmd: str, indices: VecEnvIndices, *payload) -> list[Any]:
        """Send `cmd` to the workers owning `indices`; answers come back in index order."""
        idx = list(self._get_indices(indices))
        by_worker: dict[int, list[int]] = {}
        for i in idx:
            w, j = self.where[i]
            by_worker.setdefault(w, []).append(j)
        for w, local in by_worker.items():
            self.remotes[w].send((cmd, (*payload, local)))
        answers = {w: iter(self.remotes[w].recv()) for w in by_worker}
        return [next(answers[self.where[i][0]]) for i in idx]

    def has_attr(self, attr_name: str) -> bool:
        return all(self._call("has_attr", None, attr_name))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return self._call("get_attr", indices, attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._call("set_attr", indices, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        return self._call("env_method", indices, method_name, method_args, method_kwargs)

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return self._call("is_wrapped", indices, wrapper_class)
