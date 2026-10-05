"""Labyrint - a digital version of the classic wooden tilt maze, built for AI training."""
from .game import GameState, LabyrinthGame, PhysicsConfig, Status
from .level import Level, available_levels, load_level

__all__ = ["GameState", "LabyrinthGame", "Level", "PhysicsConfig", "Status", "available_levels", "load_level"]

try:
    from gymnasium.envs.registration import register, registry

    if "Labyrint-v0" not in registry:
        register(id="Labyrint-v0", entry_point="labyrint.env:LabyrinthEnv")
except ImportError:  # gymnasium is only needed for the AI part
    pass
