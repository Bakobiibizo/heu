"""
HH-like Envelope Unit (v1.0)
============================

Core envelope unit with HH-inspired dynamics for musical applications.

Key Features:
- Fast self-amplifying transient (attack gate)
- Delayed recovery and refractory spacing (recovery gate)
- Optional inhibit gate for sharper transient cutoff
- Tonic input for sustained envelopes
- Homeostatic safety with damping
- Compressor-style lookahead for peak anticipation
- Oversampling for improved transient response

Dynamics:
---------
Gate targets use clean sigmoids:
    attack_target = sigmoid(attack_slope * (value - attack_bias))
    recovery_target = sigmoid(recovery_slope * (value - recovery_bias))

Value dynamics:
    drive = excitation + tonic
    attack_current = attack_strength * attack_gate * inhibit_gate * drive
    recovery_current = recovery_strength * recovery_gate * value
    leak_current = leak_strength * value
    dvalue/dt = attack_current - recovery_current - leak_current

Refractory uses countdown:
    new_refractory_ms = max(0, refractory_remaining_ms - dt_ms)
    if spike_condition and new_refractory_ms <= 0:
        spike = True
        new_refractory_ms = refractory_ref_ms

Lookahead (optional):
    Compressor-style lookahead that anticipates peaks and applies
    pre-emptive gain reduction to prevent saturation while preserving
    relative dynamics (accent vs non-accent).

    Configure via HEUConfig(lookahead_ms=10.0) or unit.set_lookahead(10.0)
    Use process_batch() for batch processing with lookahead.

Oversampling (optional):
    Internal oversampling for improved transient response.
    Configure via HEUConfig(oversample_factor=4) or unit.set_oversample(4)

Usage:
------
    from hh_envelope.core import HHEnvelopeUnit

    unit = HHEnvelopeUnit()

    # Run simulation (sample-by-sample)
    for excitation in signal:
        output = unit.step(excitation, tonic=0.0, dt_ms=1.0)
        print(output.envelope, output.spike_event)

    # Batch processing with lookahead
    unit.set_lookahead(lookahead_ms=10.0, ceiling=0.9)
    outputs = unit.process_batch(excitations, dt_ms=1.0)

    # Enable oversampling
    unit.set_oversample(4)  # 4x internal rate

    # Adjust parameters
    unit.set_theta(theta_total=0.5, theta_balance=-0.3)

    # Serialize/restore state
    state = unit.state_dict()
    unit.load_state_dict(state)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .config import HEUConfig, HEUParamsRef, HEUThetaBounds, SafetyConfig
from .heu_state import HEUModulators, HEUState, HEUStep, Modulators
from .integrators import exponential_decay_step
from .safety import ActivityMonitor, clamp_theta

if TYPE_CHECKING:
    from collections.abc import Sequence


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


class HHEnvelopeUnit:
    """HH-like Envelope Unit for musical dynamics.

    A continuous-time dynamical envelope generator inspired by
    Hodgkin-Huxley neuron dynamics, adapted for musical envelope shaping.

    This is a standalone primitive with zero transcription policy.
    """

    def __init__(
        self,
        params_ref: HEUParamsRef | None = None,
        theta_bounds: HEUThetaBounds | None = None,
        safety: SafetyConfig | None = None,
        config: HEUConfig | None = None,
    ):
        """Initialize the envelope unit.

        Args:
            params_ref: Reference parameter values (legacy, overridden by config)
            theta_bounds: Bounds for learnable parameters (legacy, overridden by config)
            safety: Homeostatic safety configuration (legacy, overridden by config)
            config: Unified configuration (takes precedence if provided)
        """
        # Use unified config if provided, otherwise use legacy params
        if config is not None:
            self._config: HEUConfig | None = config
            self.params_ref = config.to_params_ref()
            self.theta_bounds = config.to_theta_bounds()
            self.safety_config = config.to_safety_config()
        else:
            self._config = None
            self.params_ref = params_ref or HEUParamsRef()
            self.theta_bounds = theta_bounds or HEUThetaBounds()
            self.safety_config = safety or SafetyConfig()

        # Gate dynamics params (from unified config or defaults)
        if self._config is not None:
            self._attack_slope = self._config.attack_slope
            self._attack_bias = self._config.attack_bias
            self._recovery_slope = self._config.recovery_slope
            self._recovery_bias = self._config.recovery_bias
            self._inhibit_slope = self._config.inhibit_slope
            self._inhibit_bias = self._config.inhibit_bias
            self._enable_inhibit_gate = self._config.enable_inhibit_gate
            self._leak_strength = self._config.leak_strength
        else:
            # Defaults (attack_bias=0.5 prevents resting gate buildup, preserves accent dynamics)
            self._attack_slope = 5.0
            self._attack_bias = 0.5
            self._recovery_slope = 4.0
            self._recovery_bias = 0.2
            self._inhibit_slope = 6.0
            self._inhibit_bias = 0.5
            self._enable_inhibit_gate = False
            self._leak_strength = 0.1

        # Initialize theta values to 0 (neutral = reference values)
        self._theta_total: float = 0.0
        self._theta_balance: float = 0.0
        self._theta_attack_tau: float = 0.0
        self._theta_recovery_tau: float = 0.0

        # State (using new HEUState with refractory_remaining_ms and activity_ema)
        self._state = HEUState()
        self._current_time_ms: float = 0.0

        # Safety monitor (activity_ema is now in state, but we keep monitor for damping logic)
        self._activity_monitor = ActivityMonitor(
            tau_ms=self.safety_config.homeostasis_tau_ms,
            target=self.safety_config.activity_target,
            ceiling=self.safety_config.activity_ceiling,
            min_damp=self.safety_config.min_damping_factor,
        )

        # Lookahead configuration (compressor-style)
        if self._config is not None:
            self._lookahead_ms = self._config.lookahead_ms
            self._lookahead_mode: str = self._config.lookahead_mode
            self._lookahead_ceiling = self._config.lookahead_ceiling
            self._oversample_factor = self._config.oversample_factor
            self._integrator = self._config.integrator
        else:
            self._lookahead_ms = 0.0
            self._lookahead_mode = "peak"
            self._lookahead_ceiling = 0.95
            self._oversample_factor = 1
            self._integrator = "semi_implicit"

        # Lookahead buffer (populated during batch processing)
        self._lookahead_gain: float = 1.0  # Current lookahead gain multiplier

        # Note state for MIDI note-off detection
        self._note_active: bool = False
        self._note_off_threshold: float = (
            config.note_off_threshold if config is not None else 0.1
        )

    # --- Computed properties from theta ---

    @property
    def total_conductance(self) -> float:
        """Total conductance (attack + recovery) from theta."""
        return math.exp(self._theta_total) * self.params_ref.total_conductance_ref

    @property
    def attack_ratio(self) -> float:
        """Fraction of conductance allocated to attack (0..1)."""
        return sigmoid(self._theta_balance)

    @property
    def attack_strength(self) -> float:
        """Effective attack conductance."""
        return self.total_conductance * self.attack_ratio

    @property
    def recovery_strength(self) -> float:
        """Effective recovery conductance."""
        return self.total_conductance * (1.0 - self.attack_ratio)

    @property
    def attack_tau_ms(self) -> float:
        """Attack time constant in ms."""
        return math.exp(self._theta_attack_tau) * self.params_ref.attack_tau_ref_ms

    @property
    def recovery_tau_ms(self) -> float:
        """Recovery time constant in ms."""
        return math.exp(self._theta_recovery_tau) * self.params_ref.recovery_tau_ref_ms

    @property
    def state(self) -> HEUState:
        """Current internal state (read-only copy)."""
        return self._state.copy()

    # --- Main step function ---

    def step(
        self,
        excitation: float,
        dt_ms: float = 1.0,
        mod: Modulators | HEUModulators | None = None,
        tonic: float = 0.0,
    ) -> HEUStep:
        """Advance the unit by one timestep.

        Args:
            excitation: Input excitation (onset energy, impulse, etc.)
            dt_ms: Timestep in milliseconds (default 1.0ms = 1kHz control rate)
            mod: Optional modulators from higher layers
            tonic: Continuous low-level drive for sustained envelopes (NEW)

        Returns:
            HEUStep with envelope, spike_event, updated state, raw_value
        """
        # Apply oversampling if enabled
        if self._oversample_factor > 1:
            return self._step_oversampled(excitation, dt_ms, mod, tonic)
        else:
            return self._step_internal(excitation, dt_ms, mod, tonic)

    def _step_oversampled(
        self,
        excitation: float,
        dt_ms: float,
        mod: Modulators | HEUModulators | None,
        tonic: float,
    ) -> HEUStep:
        """Step with internal oversampling for improved transient response."""
        sub_dt = dt_ms / self._oversample_factor

        # First substep gets the full excitation, rest get none (impulse model)
        output = self._step_internal(excitation, sub_dt, mod, tonic)

        # Remaining substeps with zero excitation (let dynamics settle)
        for _ in range(1, self._oversample_factor):
            output = self._step_internal(0.0, sub_dt, mod, tonic)

        return output

    def _step_internal(
        self,
        excitation: float,
        dt_ms: float,
        mod: Modulators | HEUModulators | None,
        tonic: float,
    ) -> HEUStep:
        """Internal step implementation (single timestep, no oversampling)."""
        if mod is None:
            mod = HEUModulators()

        # Apply lookahead gain (set by process_batch or default to 1.0)
        excitation *= self._lookahead_gain

        # Apply modulation to effective parameters
        effective_attack = self.attack_strength * mod.attack_gain
        effective_recovery = self.recovery_strength * mod.recovery_gain
        effective_excitability = mod.excitability
        effective_attack_tau = self.attack_tau_ms
        effective_recovery_tau = self.recovery_tau_ms
        effective_refractory = self.params_ref.refractory_ref_ms * mod.refractory_scale

        # Get safety damping factor
        damping = self._activity_monitor.get_damping_factor()
        effective_attack *= damping
        excitation *= damping

        # Current state
        value = self._state.value
        attack_gate = self._state.attack_gate
        recovery_gate = self._state.recovery_gate
        inhibit_gate = self._state.inhibit_gate
        refractory_remaining = self._state.refractory_remaining_ms

        # --- Gate dynamics (clean sigmoid targets) ---

        # Combined drive: excitation + tonic
        drive = excitation * effective_excitability + tonic

        # Attack gate target: sigmoid based on value AND drive
        # When value is above threshold OR drive is present, attack gate opens
        attack_input = max(value, drive)  # Either sustained value or new drive
        attack_target = sigmoid(self._attack_slope * (attack_input - self._attack_bias))

        # Add positive feedback: if attack gate is already open and value elevated,
        # maintain the gate opening (creates self-sustaining transient)
        value_feedback = sigmoid(self._attack_slope * (value - self._attack_bias))
        feedback_contrib = value_feedback * attack_gate * 0.8
        attack_target = min(1.0, attack_target + feedback_contrib)

        new_attack_gate = exponential_decay_step(
            attack_gate, attack_target, effective_attack_tau, dt_ms
        )

        # Recovery gate: targets current value (delayed negative feedback)
        recovery_target = sigmoid(self._recovery_slope * (value - self._recovery_bias))
        new_recovery_gate = exponential_decay_step(
            recovery_gate, recovery_target, effective_recovery_tau, dt_ms
        )

        # Inhibit gate: closes when value is high (optional)
        if self._enable_inhibit_gate:
            # Inhibit target: closes (toward 0) when value is high
            inhibit_target = 1.0 - sigmoid(self._inhibit_slope * (value - self._inhibit_bias))
            inhibit_tau = effective_attack_tau * 4.0  # Slower than attack
            new_inhibit_gate = exponential_decay_step(
                inhibit_gate, inhibit_target, inhibit_tau, dt_ms
            )
        else:
            # Inhibit gate disabled: always fully open
            new_inhibit_gate = 1.0

        # --- Value dynamics with configurable integrator ---
        # dvalue/dt = attack_current - recovery_current - leak_current + direct_input

        def compute_derivative(v: float) -> float:
            """Compute dv/dt at a given value state."""
            # Excitatory term: attack drives value up
            atk_term = effective_attack * new_attack_gate * new_inhibit_gate * (1.0 - v) * drive
            # Recovery term: recovery drives value down
            rec_term = effective_recovery * new_recovery_gate * v
            # Leak term: passive decay
            lk_term = self._leak_strength * v
            # Direct input
            dir_in = drive * 0.55
            return atk_term - rec_term - lk_term + dir_in

        if self._integrator == "rk2":
            # RK2 (midpoint method) for better accuracy
            k1 = compute_derivative(value)
            k2 = compute_derivative(value + 0.5 * dt_ms * k1)
            new_value = value + dt_ms * k2
        elif self._integrator == "euler":
            # Explicit Euler (simplest)
            new_value = value + dt_ms * compute_derivative(value)
        else:  # semi_implicit (default)
            # Semi-implicit: gates updated first, then value
            new_value = value + dt_ms * compute_derivative(value)

        # Compute final derivative for spike detection check
        dvalue_dt = compute_derivative(new_value)

        # Soft clamp to prevent extreme values
        new_value = max(-0.1, min(1.5, new_value))

        # --- Refractory countdown ---
        new_refractory = max(0.0, refractory_remaining - dt_ms)

        # --- Spike detection ---
        spike_event = False
        effective_threshold = self.params_ref.spike_threshold + mod.threshold_offset
        effective_threshold = max(0.1, min(1.5, effective_threshold))
        if new_refractory <= 0.0 and new_value >= effective_threshold:
            # Additional check: must be rising
            if dvalue_dt > 0:
                spike_event = True
                new_refractory = effective_refractory

        # --- Update activity EMA in state ---
        alpha = dt_ms / self.safety_config.homeostasis_tau_ms
        new_activity_ema = self._state.activity_ema + alpha * (
            abs(new_value) - self._state.activity_ema
        )

        # --- Update state ---
        self._state = HEUState(
            value=new_value,
            attack_gate=new_attack_gate,
            recovery_gate=new_recovery_gate,
            inhibit_gate=new_inhibit_gate,
            refractory_remaining_ms=new_refractory,
            activity_ema=new_activity_ema,
        )

        # Update time
        self._current_time_ms += dt_ms

        # Compute output envelope (normalized to 0..1)
        envelope = max(0.0, min(1.0, new_value))

        # Update activity monitor (for damping calculation)
        self._activity_monitor.update(envelope, dt_ms)

        # --- Note-off detection for MIDI export ---
        note_off_event = False
        if self._note_active and envelope < self._note_off_threshold:
            note_off_event = True
            self._note_active = False
        elif spike_event:
            # Spike triggers note-on
            self._note_active = True

        return HEUStep(
            envelope=envelope,
            spike_event=spike_event,
            note_off_event=note_off_event,
            state=self._state.copy(),
            raw_value=new_value,
        )

    def reset(self) -> None:
        """Reset the unit to initial state."""
        self._state = HEUState()
        self._current_time_ms = 0.0
        self._activity_monitor.reset()
        self._lookahead_gain = 1.0
        self._note_active = False

    # --- Batch processing with lookahead ---

    def process_batch(
        self,
        excitations: Sequence[float],
        dt_ms: float = 1.0,
        tonic: float | Sequence[float] = 0.0,
        mod: Modulators | HEUModulators | None = None,
    ) -> list[HEUStep]:
        """Process a batch of excitations with lookahead support.

        This method enables compressor-style lookahead by analyzing future
        excitations to anticipate peaks and apply pre-emptive gain reduction.

        Args:
            excitations: Sequence of excitation values
            dt_ms: Timestep in milliseconds
            tonic: Constant tonic or sequence of tonic values
            mod: Optional modulators

        Returns:
            List of HEUStep outputs for each timestep
        """
        n_samples = len(excitations)

        # Handle tonic as constant or sequence
        if isinstance(tonic, (int, float)):
            tonic_seq = [float(tonic)] * n_samples
        else:
            tonic_seq = list(tonic)

        # Compute lookahead samples
        lookahead_samples = int(self._lookahead_ms / dt_ms) if dt_ms > 0 else 0

        # Pre-compute lookahead gains if enabled
        if lookahead_samples > 0:
            lookahead_gains = self._compute_lookahead_gains(excitations, lookahead_samples, dt_ms)
        else:
            lookahead_gains = [1.0] * n_samples

        # Process each sample
        outputs: list[HEUStep] = []
        for i in range(n_samples):
            # Apply lookahead gain to excitation
            self._lookahead_gain = lookahead_gains[i]
            output = self.step(
                excitation=excitations[i],
                dt_ms=dt_ms,
                tonic=tonic_seq[i],
                mod=mod,
            )
            outputs.append(output)

        return outputs

    def _compute_lookahead_gains(
        self,
        excitations: Sequence[float],
        lookahead_samples: int,
        dt_ms: float,
    ) -> list[float]:
        """Compute lookahead gain for each sample.

        Looks ahead to find the peak excitation and computes a gain
        that would bring the resulting envelope to the ceiling.

        Args:
            excitations: Sequence of excitation values
            lookahead_samples: Number of samples to look ahead
            dt_ms: Timestep in milliseconds

        Returns:
            List of gain values for each sample
        """
        n = len(excitations)
        gains = [1.0] * n

        for i in range(n):
            # Find peak in lookahead window
            end_idx = min(i + lookahead_samples + 1, n)
            window = excitations[i:end_idx]

            if self._lookahead_mode == "peak":
                peak = max(abs(x) for x in window) if window else 0.0
            else:  # rms
                peak = math.sqrt(sum(x * x for x in window) / len(window)) if window else 0.0

            # Compute gain to bring peak to ceiling
            # The direct_input term in step() is drive * 0.55, so we estimate
            # the expected envelope response
            if peak > 0:
                expected_response = peak * 0.55 + peak * self.attack_strength * 0.5
                if expected_response > self._lookahead_ceiling:
                    gains[i] = self._lookahead_ceiling / expected_response
                    gains[i] = max(0.1, min(1.0, gains[i]))  # Clamp gain

        # Smooth the gain to avoid abrupt changes
        smoothed = self._smooth_gains(gains, lookahead_samples)
        return smoothed

    def _smooth_gains(self, gains: list[float], window: int) -> list[float]:
        """Smooth gain changes to avoid pumping artifacts.

        Uses a simple attack/release envelope follower.
        """
        if window <= 0:
            return gains

        smoothed = gains.copy()
        attack_coef = 0.9  # Fast attack
        release_coef = 0.1  # Slow release

        for i in range(1, len(gains)):
            if gains[i] < smoothed[i - 1]:
                # Attack (gain decreasing = more compression)
                smoothed[i] = attack_coef * gains[i] + (1 - attack_coef) * smoothed[i - 1]
            else:
                # Release (gain increasing = less compression)
                smoothed[i] = release_coef * gains[i] + (1 - release_coef) * smoothed[i - 1]

        return smoothed

    # --- Parameter access ---

    def set_theta(
        self,
        theta_total: float | None = None,
        theta_balance: float | None = None,
        theta_attack_tau: float | None = None,
        theta_recovery_tau: float | None = None,
    ) -> None:
        """Set theta parameters with bounds clamping.

        Args:
            theta_total: Log-scale total conductance
            theta_balance: Sigmoid-scale attack/recovery balance
            theta_attack_tau: Log-scale attack time constant
            theta_recovery_tau: Log-scale recovery time constant
        """
        if theta_total is not None:
            self._theta_total = clamp_theta(theta_total, self.theta_bounds.theta_total)
        if theta_balance is not None:
            self._theta_balance = clamp_theta(theta_balance, self.theta_bounds.theta_balance)
        if theta_attack_tau is not None:
            self._theta_attack_tau = clamp_theta(
                theta_attack_tau, self.theta_bounds.theta_attack_tau
            )
        if theta_recovery_tau is not None:
            self._theta_recovery_tau = clamp_theta(
                theta_recovery_tau, self.theta_bounds.theta_recovery_tau
            )

    def get_theta(self) -> dict[str, float]:
        """Get current theta parameters."""
        return {
            "theta_total": self._theta_total,
            "theta_balance": self._theta_balance,
            "theta_attack_tau": self._theta_attack_tau,
            "theta_recovery_tau": self._theta_recovery_tau,
        }

    def set_theta_from_dict(self, theta_dict: dict[str, float]) -> None:
        """Set theta parameters from a dictionary."""
        self.set_theta(
            theta_total=theta_dict.get("theta_total"),
            theta_balance=theta_dict.get("theta_balance"),
            theta_attack_tau=theta_dict.get("theta_attack_tau"),
            theta_recovery_tau=theta_dict.get("theta_recovery_tau"),
        )

    # --- Serialization methods (v1.0 spec) ---

    def state_dict(self) -> dict:
        """Serialize full unit state to dictionary.

        Returns:
            Dictionary containing state, theta values, and timing
        """
        return {
            "state": self._state.to_dict(),
            "theta": self.get_theta(),
            "current_time_ms": self._current_time_ms,
            "activity_monitor_ema": self._activity_monitor.activity_ema,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore unit state from dictionary.

        Args:
            d: Dictionary from state_dict()
        """
        if "state" in d:
            self._state = HEUState.from_dict(d["state"])
        if "theta" in d:
            self.set_theta_from_dict(d["theta"])
        if "current_time_ms" in d:
            self._current_time_ms = d["current_time_ms"]
        if "activity_monitor_ema" in d:
            self._activity_monitor._activity_ema = d["activity_monitor_ema"]

    def params_dict(self) -> dict:
        """Get all learnable parameters as dictionary.

        Returns:
            Dictionary of theta parameters and computed values
        """
        return {
            "theta": self.get_theta(),
            "computed": {
                "total_conductance": self.total_conductance,
                "attack_strength": self.attack_strength,
                "recovery_strength": self.recovery_strength,
                "attack_tau_ms": self.attack_tau_ms,
                "recovery_tau_ms": self.recovery_tau_ms,
            },
        }

    def set_params_dict(self, d: dict) -> None:
        """Set learnable parameters from dictionary.

        Args:
            d: Dictionary with "theta" key containing theta parameters
        """
        if "theta" in d:
            self.set_theta_from_dict(d["theta"])

    # --- Legacy properties for compatibility ---

    @property
    def activity_ema(self) -> float:
        """Current activity EMA from safety monitor."""
        return self._activity_monitor.activity_ema

    def should_freeze_learning(self) -> bool:
        """Check if learning should be frozen due to high activity."""
        return self._activity_monitor.should_freeze_learning()

    # --- Lookahead and oversampling configuration ---

    def set_lookahead(
        self,
        lookahead_ms: float,
        mode: str = "peak",
        ceiling: float = 0.95,
    ) -> None:
        """Configure lookahead at runtime.

        Args:
            lookahead_ms: Lookahead time in ms (0 to disable)
            mode: Detection mode ("peak" or "rms")
            ceiling: Target ceiling for gain control
        """
        self._lookahead_ms = max(0.0, lookahead_ms)
        self._lookahead_mode = mode if mode in ("peak", "rms") else "peak"
        self._lookahead_ceiling = max(0.1, min(1.0, ceiling))

    def set_oversample(self, factor: int) -> None:
        """Configure oversampling at runtime.

        Args:
            factor: Oversampling factor (1 = no oversampling, max 16)
        """
        self._oversample_factor = max(1, min(16, factor))

    @property
    def lookahead_ms(self) -> float:
        """Current lookahead time in ms."""
        return self._lookahead_ms

    @property
    def oversample_factor(self) -> int:
        """Current oversampling factor."""
        return self._oversample_factor
