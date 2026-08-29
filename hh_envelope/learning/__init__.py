"""
Learning module: Loss functions and optimization for envelope fitting.

Priority: P2

Components:
    loss: Loss functions (MSE, smoothness penalty, combined)
    optim_cmaes: CMA-ES optimizer wrapper
    optim_random: Random/grid search baseline

Learning scope (v0.1):
- Learn only: theta_total, theta_balance, theta_recovery_tau, optionally theta_attack_tau
- Update only at spike events or phrase boundaries, or at 10-50 Hz max
- Freeze learning if activity_ema exceeds hard ceiling
"""

from .loss import combined_loss, envelope_mse, smoothness_penalty, temporal_alignment_loss
from .optim_cmaes import CMAESOptimizer, OptimizeResult
from .optim_random import GridSearchOptimizer, RandomSearchOptimizer

__all__ = [
    "envelope_mse",
    "smoothness_penalty",
    "combined_loss",
    "temporal_alignment_loss",
    "CMAESOptimizer",
    "OptimizeResult",
    "RandomSearchOptimizer",
    "GridSearchOptimizer",
]
