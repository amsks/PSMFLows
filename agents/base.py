"""
Base agent and utility functions shared across FB variants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Soft / hard parameter update helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak update: θ_target ← τ·θ_source + (1-τ)·θ_target."""
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)


@torch.no_grad()
def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Copy source parameters into target."""
    target.load_state_dict(source.state_dict())


# ──────────────────────────────────────────────────────────────────────────────
# BaseAgent
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Minimal interface every FB agent must implement."""

    @abstractmethod
    def update(self, batch: Dict[str, Tensor], step: int) -> Dict[str, float]:
        """Run one gradient update step on *batch*.
        Returns a dict of scalar metrics for logging."""

    @abstractmethod
    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        """Select an action given observation *obs* and goal embedding *z*."""

    @abstractmethod
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        """Infer goal embedding z from (obs, reward) pairs via the linear
        estimator z = project(sum(r * B(obs)))."""

    def reset(self) -> None:
        """Called at the start of each eval episode. Default: no-op.
        Stateful planners (e.g. TD-MPC2) override this to clear per-episode state."""
        return None

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Any = None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))