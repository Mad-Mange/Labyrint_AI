# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A digital version of the classic wooden tilt maze (reference photo: `pictures/Labyrint.png`, which exists only locally: it is a product photo, so it is gitignored and was removed from the public history), built so that a neural network can later be trained to play it with reinforcement learning. Phase 1 (game, renderer, Gymnasium env, scripted baseline) is done. Phase 2 is RL training with stable-baselines3 PPO (`train.py`; CUDA torch, the machine has an RTX 3080).

The user writes in Swedish. UI text is Swedish by default with an English translation (`--lang en`, see **UI text** below). `README.md` is English because the GitHub repo (Mad-Mange/Labyrint_AI) is public. Code identifiers and comments are English.

## Commands

Use the project venv (Python 3.12, chosen over the system Python 3.14 for torch compatibility). From Git Bash use `./.venv/Scripts/python.exe`; from PowerShell use `.venv\Scripts\python`.

```bash
.venv/Scripts/python.exe play.py                       # play (mouse / arrows); --a to watch the baseline
.venv/Scripts/python.exe play.py --level pinnar        # the harder level (pins along the edges)
.venv/Scripts/python.exe play.py --lang en             # English UI (also for dashboard.py)
.venv/Scripts/python.exe -m pytest                     # all tests (~1 s)
.venv/Scripts/python.exe -m pytest tests/test_game.py::test_autopilot_finishes   # single test
.venv/Scripts/python.exe -m labyrint.autopilot --episodes 20   # headless baseline: success rate, time, ticks/s
.venv/Scripts/python.exe train.py --name ppo           # train PPO (16 subprocess envs) -> runs/ppo/best_model.zip
.venv/Scripts/python.exe -m labyrint.agent runs/ppo/best_model.zip --episodes 20   # measure a trained model
.venv/Scripts/python.exe play.py --ai [model.zip]      # watch a model (default models/labyrint_ai.zip)
.venv/Scripts/python.exe play.py --live [runs/<name>]  # watch a run while it trains (default: newest run)
.venv/Scripts/python.exe dashboard.py [runs/<name>]    # live matplotlib graphs of a run
.venv/Scripts/tensorboard.exe --logdir runs
.venv/Scripts/python.exe -c "from labyrint.level import load_level, validate_level; print(validate_level(load_level('classic')))"
```

Rendering without a window (screenshots, `rgb_array`): set `SDL_VIDEODRIVER=dummy`. No linter or formatter is configured.

## Architecture

The core idea: **`LabyrinthGame` (`labyrint/game.py`) is a pure headless simulation**, and everything else is layered on top without changing it:

- `render.py` only *reads* the game (human play, autopilot and AI all use the same draw path). It never calls `convert()`, so it works without a display. `board_array()` gives pixels for the env.
- `env.py` wraps the game as Gymnasium `Labyrint-v0` (registered in `labyrint/__init__.py`, and only if gymnasium is installed).
- `autopilot.py` is a hand-written path-following controller. It is used as a solvability check and as the time to beat (`baseline_seconds(level)`, ≈38 s on both levels).
- `agent.py` (`NeuralPilot`) runs a trained SB3 model as a controller with the same `act(game)` interface as the autopilot. torch/SB3 are imported lazily so the game runs without them.
- `train.py` (top level) trains with SB3: `SubprocVecEnv` + `VecNormalize` (rewards only, so models need no stats to play), curriculum via `random_start`, and an eval callback that plays full games from START and keeps `best_model.zip` by (success rate, mean time). `runs/` is gitignored; a model worth keeping is copied to `models/labyrint_ai.zip`.
- `play.py` holds the human input loop (fixed-timestep accumulator, fall animation, attempts/best-time HUD).
- **Live training views** are decoupled from the training through files in `runs/<name>/`. `LiveCallback` in `train.py` appends every logger dump to `progress.jsonl` (a `JsonlWriter` added to SB3's logger), which `dashboard.py` tails. Every 3 s it also writes `live_policy.pt` + `live.json` via `write_live_snapshot` (temp file + `os.replace`; it skips a snapshot if Windows reports the file busy). `LivePilot` in `agent.py` reloads the snapshot at the start of each attempt in `play.py --live`. In live mode an attempt also ends after `STALL_SECONDS` without progress, like in the env.
- **Which level a model plays:** `play.py --live`, `play.py --ai runs/<name>/...`, `labyrint.agent` and `dashboard.py` read the level from the run's `args.json` (`run_level` in `agent.py`), falling back to `classic` (e.g. for `models/labyrint_ai.zip`, which is trained on `classic`). `trana_ai_live.bat` starts the windows before `train.py` has written `args.json`, so `play.py --live` waits up to `LEVEL_WAIT` for it and the dashboard picks the level up when it appears.
- `spela_ai.bat` / `trana_ai_live.bat` are double-click launchers for the user (`trana_ai_live.bat` trains on `pinnar`; extra flags such as `--level classic` override it). They are saved in **code page 850, not UTF-8**: cmd misreads batch files after `chcp 65001` when they contain å/ä/ö. Regenerate them with `encoding="cp850", newline="\r\n"`.
- A training started from a console window dies when that window is closed (the Intel Fortran runtime prints `forrtl: error (200): ... window-CLOSE event`). To run one detached, start it hidden via `Win32_Process.Create` with `ShowWindow=0`.

Conventions that span files:

- **Units:** millimetres and seconds, with x right and **y down** (screen orientation). Tilt is normalised to [-1, 1] per axis (1 = `max_tilt_deg`). Positive tilt accelerates the ball in the positive direction. An action is a *target* tilt; the board moves towards it at a limited speed (`tilt_speed_deg`), like turning the knobs.
- **Timing:** `game.step()` = one tick of 1/60 s, split into 4 physics substeps. The env uses `frame_skip=2`, so the AI acts 30 times per second.
- **Progress:** the black guide line (`Path`, parameterised by arc length `s`) is how progress is measured, and it drives the reward. `_update_progress` projects the ball onto the path **only within ±60 mm of the previous `s`**. Without that window, the projection would jump through a wall to a neighbouring lap of the spiral. Keep this windowing if you touch progress or reward code.
- **Holes:** holes are numbered by their order in the level JSON, which must follow the path (`validate_level` checks this). The rim pulls the ball towards the hole centre once the ball's centre is within `hole_radius`. The ball falls when its centre is within `fall_fraction * hole_radius`.
- **Walls:** walls are oriented rectangles (`Wall`). `Level.walls` are the visible strips. `Level.frame` adds invisible collision slabs outside the board, and `all_walls` combines both.
- **Levels:** `levels/*.json` path control points are smoothed with Chaikin (`path_smoothing`), and validation runs on the *smoothed* path. After editing a level, run `validate_level` and the autopilot.
- **Edge channels:** a hole whose centre is more than `ball_radius + hole_radius` (14 mm) from a wall face leaves a channel where a ball rolling along the wall passes the hole without falling. On `classic` the edge holes sit 15–16 mm from the face, and the trained AI used those channels to finish in 7 s. `pinnar` is `classic` plus one pin per edge hole, reaching from the face to the hole rim (`test_pins_block_sneaking_along_the_edge`). An RL agent will find any such channel, so close them in new levels.
- **`clone()`:** this does a shallow `__dict__` copy (the level is shared; the rng is copied explicitly). If you add mutable state to `LabyrinthGame`, copy it explicitly in `clone()`. `GameState` is a frozen dataclass compared for equality in determinism tests.
- **Observation layout:** `observe(game)` in `env.py` (`OBS_SIZE` = 36, all values clipped to [-1, 1]) is the contract with trained models; both the env and `NeuralPilot` use it. `NeuralPilot` re-decides every `frame_skip` ticks like the env (`test_neural_pilot_replays_env_trajectory`). Changing it, `frame_skip` or `PhysicsConfig` invalidates any trained agents. The F1 debug overlay in the renderer draws the same sensors the observation uses.
- **Physics changes:** physics changes can break `test_autopilot_finishes` and shift the baseline time. Re-run the autopilot benchmark after tuning.
- **UI text:** write UI strings in Swedish wrapped in `t()` from `labyrint/lang.py`, and add the English text to `EN` there. The Swedish text is the key, and templates are translated before `.format()`. Level names and controller names are translated where they are shown. `tests/test_lang.py` finds every `t("...")` literal in `render.py`, `play.py` and `dashboard.py` and fails if a translation is missing. Console output and `--help` texts are not translated.
- **README pictures:** `docs/game.png` (English game window, AI on `pinnar`) and `docs/training.png` (`dashboard.py <run> --lang en --save docs/training.png`). Keep all text in them English, including the run folder name in the graph title.
