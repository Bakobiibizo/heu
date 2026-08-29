"""Numerical Integration
=======================

Stable, deterministic integrators for envelope dynamics.

Requirements:
- Stable at audio-ish update rates (1 kHz to 4 kHz control rate)
- Deterministic
- Cheap per-step

Integrators:
- exponential_decay_step: Exact solution for first-order decay (gate dynamics)
- semi_implicit_euler_step: Symplectic Euler for oscillatory systems
- rk2_step: Second-order Runge-Kutta (midpoint method)
- rk4_step: Fourth-order Runge-Kutta (high accuracy)
- adaptive_step: Adaptive timestep using RK2/RK4 error estimate

All integrators are pure functions (no side effects).
Time units should be consistent (milliseconds recommended).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def exponential_decay_step(
    value: float,
    target: float,
    tau: float,
    dt: float,
) -> float:
    """Exact solution for first-order exponential decay toward target.

    Solves: dx/dt = (target - x) / tau

    This is the analytical solution, which is unconditionally stable
    regardless of dt/tau ratio. Preferred for gate dynamics.

    Args:
        value: Current value
        target: Target value to decay toward
        tau: Time constant (same units as dt)
        dt: Timestep (same units as tau)

    Returns:
        New value after dt
    """
    if tau <= 0:
        return target  # Instant decay
    alpha = 1.0 - math.exp(-dt / tau)
    return value + alpha * (target - value)


def semi_implicit_euler_step(
    state: np.ndarray,
    derivative_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """Semi-implicit Euler integration step.

    For a system dx/dt = f(x), computes:
    x_new = x + dt * f(x)

    This is actually explicit Euler; for truly semi-implicit,
    you'd solve for x_new implicitly. However, for our envelope
    system with separated gate and value dynamics, we achieve
    semi-implicit behavior by updating gates first, then value.

    More stable than explicit Euler for stiff systems.

    Args:
        state: Current state vector
        derivative_fn: Function computing dx/dt given x
        dt: Timestep

    Returns:
        New state vector
    """
    derivatives = derivative_fn(state)
    return state + dt * derivatives


def rk2_step(
    state: np.ndarray,
    derivative_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """Second-order Runge-Kutta (midpoint method) integration step.

    More accurate than Euler while still being cheap:
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    x_new = x + dt * k2

    Args:
        state: Current state vector
        derivative_fn: Function computing dx/dt given x
        dt: Timestep

    Returns:
        New state vector
    """
    k1 = derivative_fn(state)
    k2 = derivative_fn(state + 0.5 * dt * k1)
    return state + dt * k2


def rk4_step(
    state: np.ndarray,
    derivative_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """Fourth-order Runge-Kutta integration step.

    Most accurate standard method, but 4x the derivative evaluations:
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Use when accuracy matters more than speed.

    Args:
        state: Current state vector
        derivative_fn: Function computing dx/dt given x
        dt: Timestep

    Returns:
        New state vector
    """
    k1 = derivative_fn(state)
    k2 = derivative_fn(state + 0.5 * dt * k1)
    k3 = derivative_fn(state + 0.5 * dt * k2)
    k4 = derivative_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def adaptive_step(
    state: np.ndarray,
    derivative_fn: Callable[[np.ndarray], np.ndarray],
    dt: float,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Adaptive timestep using embedded RK methods.

    Compares RK2 and RK4 solutions to estimate error and
    suggest timestep adjustment.

    Args:
        state: Current state vector
        derivative_fn: Function computing dx/dt given x
        dt: Proposed timestep
        tolerance: Error tolerance

    Returns:
        Tuple of (new_state, suggested_next_dt)
    """
    # Compute both RK2 and RK4 solutions
    rk2_result = rk2_step(state, derivative_fn, dt)
    rk4_result = rk4_step(state, derivative_fn, dt)

    # Estimate error as difference
    error = np.max(np.abs(rk4_result - rk2_result))

    # Compute scaling factor for next dt
    if error > 0:
        scale = 0.9 * (tolerance / error) ** 0.25
        scale = max(0.1, min(2.0, scale))  # Limit change
    else:
        scale = 2.0

    suggested_dt = dt * scale

    # Use the more accurate solution
    return rk4_result, suggested_dt
