"""
Homeostatic Safety (v1.0)
=========================

Safety mechanisms to prevent runaway dynamics.

Philosophy:
-----------
- Safety is "do no harm" - it only constrains, never adds energy
- Damping scales conductance or input gain, doesn't change structure
- Learning freezes if system is saturated (prevents fitting to pathology)

Components:
-----------
1. ActivityMonitor: Tracks activity EMA and provides damping factor
2. clamp_theta(): Clamp theta values to bounds
3. apply_bounds_to_all_theta(): Apply bounds to all theta parameters

Time Units:
-----------
All time parameters use milliseconds internally (tau_ms, dt_ms).

Usage:
------
    from hh_envelope.core.safety import ActivityMonitor, clamp_theta

    monitor = ActivityMonitor(tau_ms=500, target=0.3, ceiling=0.8, min_damp=0.1)

    # In simulation loop
    damping = monitor.get_damping_factor()
    effective_input = input * damping
    output = unit.step(effective_input, dt_ms=1.0)
    monitor.update(output.envelope, dt_ms=1.0)

    # In learning loop
    if monitor.should_freeze_learning():
        print("Activity too high, skipping learning update")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import HEUThetaBounds


class ActivityMonitor:
    """Tracks activity level and provides damping for homeostatic safety.

    Uses exponential moving average (EMA) of envelope magnitude to
    detect sustained high activity and apply proportional damping.
    """

    def __init__(
        self,
        tau_ms: float = 500.0,
        target: float = 0.3,
        ceiling: float = 0.8,
        min_damp: float = 0.1,
    ):
        """Initialize the activity monitor.

        Args:
            tau_ms: EMA time constant in milliseconds
            target: Target activity level (damping starts above this)
            ceiling: Hard ceiling (freeze learning above this)
            min_damp: Minimum damping factor (never damp below this)
        """
        self.tau_ms = tau_ms
        self.target = target
        self.ceiling = ceiling
        self.min_damp = min_damp

        # Initialize EMA at target (neutral starting point)
        self._activity_ema = target

    @property
    def activity_ema(self) -> float:
        """Current exponential moving average of activity."""
        return self._activity_ema

    def update(self, envelope: float, dt_ms: float) -> float:
        """Update the activity EMA with new envelope value.

        Args:
            envelope: Current envelope value
            dt_ms: Timestep in milliseconds

        Returns:
            Updated activity EMA
        """
        # EMA update: ema += (dt/tau) * (sample - ema)
        alpha = dt_ms / self.tau_ms
        # Use absolute value to track magnitude
        self._activity_ema += alpha * (abs(envelope) - self._activity_ema)
        return self._activity_ema

    def get_damping_factor(self) -> float:
        """Compute damping factor based on current activity.

        Returns:
            Damping factor in [min_damp, 1.0]
            - 1.0 when activity <= target (no damping)
            - Decreases linearly toward min_damp as activity approaches ceiling
            - min_damp when activity >= ceiling
        """
        if self._activity_ema <= self.target:
            return 1.0
        elif self._activity_ema >= self.ceiling:
            return self.min_damp
        else:
            # Linear interpolation between target and ceiling
            # damping = target / activity_ema, clamped
            ratio = self.target / self._activity_ema
            return max(self.min_damp, min(1.0, ratio))

    def should_freeze_learning(self) -> bool:
        """Check if learning should be frozen due to high activity.

        Returns:
            True if activity exceeds ceiling
        """
        return self._activity_ema > self.ceiling

    def reset(self) -> None:
        """Reset the activity EMA to target."""
        self._activity_ema = self.target


def clamp_theta(theta: float, bounds: tuple[float, float]) -> float:
    """Clamp a theta value to specified bounds.

    Args:
        theta: Value to clamp
        bounds: Tuple of (lower_bound, upper_bound)

    Returns:
        Clamped value
    """
    lower, upper = bounds
    return max(lower, min(upper, theta))


def apply_bounds_to_all_theta(
    theta_dict: dict[str, float],
    bounds: "HEUThetaBounds",
) -> dict[str, float]:
    """Apply bounds clamping to all theta parameters.

    Args:
        theta_dict: Dictionary of theta parameter values
        bounds: HEUThetaBounds with bounds for each parameter

    Returns:
        New dictionary with clamped values
    """
    return {
        "theta_total": clamp_theta(theta_dict.get("theta_total", 0.0), bounds.theta_total),
        "theta_balance": clamp_theta(theta_dict.get("theta_balance", 0.0), bounds.theta_balance),
        "theta_attack_tau": clamp_theta(
            theta_dict.get("theta_attack_tau", 0.0), bounds.theta_attack_tau
        ),
        "theta_recovery_tau": clamp_theta(
            theta_dict.get("theta_recovery_tau", 0.0), bounds.theta_recovery_tau
        ),
    }
