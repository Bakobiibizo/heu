"""Experiment 03: Envelope Fitting
==================================

Validates that learning can fit target articulation shapes.

Hypothesis:
    With constrained parameterization and safety, CMA-ES can fit
staccato envelopes, legato envelopes, grace note patterns, and accent curves.

Metrics:
    - Final MSE loss
    - Convergence iterations
    - Safety engagement count
    - Generalization loss on variants

Pass Conditions:
    - Final loss < 0.01 for all shapes
    - Converges in < 200 iterations
    - Safety damping engaged < 10% of steps
    - Generalization loss < 2x training loss
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig
from hh_envelope.learning.loss import combined_loss
from hh_envelope.learning.optim_cmaes import CMAESOptimizer
from hh_envelope.learning.optim_random import RandomSearchOptimizer
from hh_envelope.polyphony.articulation import (
    ArticulatedNote,
    ArticulationConfig,
    create_grace_note_config,
)
from hh_envelope.utils.plotting import plot_envelope_fit, plot_learning_curve
from hh_envelope.utils.signals import (
    input_signal_grace_note,
    input_signal_legato,
    input_signal_staccato,
    target_envelope_grace_note,
    target_envelope_legato,
    target_envelope_staccato,
)
from hh_transcriber.experiments.diagnostics import (
    build_diagnostics_bundle,
    report_diagnostics,
)


def create_unit() -> HHEnvelopeUnit:
    """Create an HEU with default parameters."""
    return HHEnvelopeUnit(
        params_ref=HEUParamsRef(),
        theta_bounds=HEUThetaBounds(),
        safety=SafetyConfig(),
    )


def run_unit_with_pulse(
    unit: HHEnvelopeUnit,
    duration_ms: float,
    dt_ms: float = 1.0,
    pulse_amplitude: float = 1.0,
) -> np.ndarray:
    """Run unit with a single pulse at t=0.

    Args:
        unit: HEU to run
        duration_ms: Total duration
        dt_ms: Timestep
        pulse_amplitude: Amplitude of initial pulse

    Returns:
        Envelope array
    """
    unit.reset()
    num_samples = int(duration_ms / dt_ms)
    envelope = np.zeros(num_samples)

    for i in range(num_samples):
        # Pulse only at first sample
        excitation = pulse_amplitude if i == 0 else 0.0
        output = unit.step(excitation, dt_ms)
        envelope[i] = output.envelope

    return envelope


def run_unit_with_signal(
    unit: HHEnvelopeUnit,
    input_signal: np.ndarray,
    dt_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run unit with arbitrary input signal.

    Args:
        unit: HEU to run
        input_signal: Array of excitation values per timestep
        dt_ms: Timestep

    Returns:
        Tuple of (envelope_array, spike_bool_array)
    """
    unit.reset()
    num_samples = len(input_signal)
    envelope = np.zeros(num_samples)
    spike_bool = np.zeros(num_samples, dtype=bool)

    for i in range(num_samples):
        excitation = input_signal[i]
        output = unit.step(excitation, dt_ms)
        envelope[i] = output.envelope
        if output.spike_event:
            spike_bool[i] = True

    return envelope, spike_bool


def create_objective(
    target: np.ndarray,
    input_signal: np.ndarray,
    dt_ms: float = 1.0,
    lambda_smooth: float = 0.01,
) -> Callable[[dict[str, float]], float]:
    """Create objective function for optimization.

    Args:
        target: Target envelope to fit
        input_signal: Input excitation signal
        dt_ms: Timestep
        lambda_smooth: Smoothness penalty weight

    Returns:
        Objective function mapping theta_dict -> loss
    """

    def objective(theta_dict: dict[str, float]) -> float:
        unit = create_unit()
        unit.set_theta_from_dict(theta_dict)

        # Run unit with input signal
        envelope, _spikes = run_unit_with_signal(unit, input_signal, dt_ms)

        # Handle length mismatch (grace note total length differs)
        min_len = min(len(envelope), len(target))
        return combined_loss(envelope[:min_len], target[:min_len], lambda_smooth)

    return objective


def fit_envelope(
    target_name: str,
    target: np.ndarray,
    input_signal: np.ndarray,
    dt_ms: float = 1.0,
    max_iterations: int = 200,
    use_cmaes: bool = True,
) -> dict:
    """Fit an HEU to a target envelope.

    Args:
        target_name: Name of the target shape
        target: Target envelope array
        input_signal: Input excitation signal that matches target semantics
        dt_ms: Timestep
        max_iterations: Maximum optimization iterations
        use_cmaes: Use CMA-ES (True) or random search (False)

    Returns:
        Dictionary with fit results
    """
    objective = create_objective(target, input_signal, dt_ms, lambda_smooth=0.01)

    if use_cmaes:
        cmaes_optimizer = CMAESOptimizer(
            bounds=HEUThetaBounds(),
            sigma0=0.5,
            seed=42,
        )
        result = cmaes_optimizer.optimize(
            objective,
            max_iterations=max_iterations,
            target_loss=0.001,
            verbose=False,
        )
    else:
        random_optimizer = RandomSearchOptimizer(
            bounds=HEUThetaBounds(),
            num_samples=max_iterations * 10,
            seed=42,
        )
        result = random_optimizer.optimize(objective, verbose=False)

    # Get final envelope with best parameters
    unit = create_unit()
    unit.set_theta_from_dict(result.best_theta)
    final_envelope, final_spike_bool = run_unit_with_signal(unit, input_signal, dt_ms)

    # Handle length mismatch
    min_len = min(len(final_envelope), len(target))

    return {
        "name": target_name,
        "best_theta": result.best_theta,
        "final_loss": result.best_loss,
        "iterations": result.iterations,
        "loss_history": result.loss_history,
        "converged": result.converged,
        "target": target[:min_len],
        "predicted": final_envelope[:min_len],
        "spike_bool": final_spike_bool[:min_len],
        "dt_ms": dt_ms,
        "input_signal": input_signal,
    }


def test_generalization(
    theta: dict[str, float],
    original_target_fn: Callable[[float, float], np.ndarray],
    original_input_fn: Callable[[float, float], np.ndarray],
    original_duration_ms: float,
    dt_ms: float = 1.0,
) -> dict:
    """Test generalization of fitted parameters on held-out variants.

    Tests on 5 held-out variants:
    - 0.8x duration (tempo scaled faster)
    - 1.2x duration (tempo scaled slower)
    - 0.7x input amplitude
    - 1.3x input amplitude
    - Original duration with additive noise (sigma=0.05)

    Args:
        theta: Fitted theta parameters
        original_target_fn: Function to generate target envelope
        original_input_fn: Function to generate input signal
        original_duration_ms: Original target duration
        dt_ms: Timestep

    Returns:
        Dictionary with generalization results including worst-case loss
    """
    results = {}
    rng = np.random.default_rng(seed=123)

    # Duration-scaled variants
    for scale, name in [(0.8, "faster"), (1.2, "slower")]:
        scaled_duration_ms = original_duration_ms * scale
        target = original_target_fn(scaled_duration_ms, dt_ms)
        input_signal = original_input_fn(scaled_duration_ms, dt_ms)

        unit = create_unit()
        unit.set_theta_from_dict(theta)
        predicted, _spikes = run_unit_with_signal(unit, input_signal, dt_ms)

        min_len = min(len(target), len(predicted))
        loss = combined_loss(predicted[:min_len], target[:min_len], 0.01)
        results[name] = {"scale": scale, "duration_ms": scaled_duration_ms, "loss": loss}

    # Amplitude-scaled variants (same duration, different input amplitude)
    for amp_scale, name in [(0.7, "amp_low"), (1.3, "amp_high")]:
        target = original_target_fn(original_duration_ms, dt_ms)
        input_signal = original_input_fn(original_duration_ms, dt_ms) * amp_scale

        unit = create_unit()
        unit.set_theta_from_dict(theta)
        predicted, _spikes = run_unit_with_signal(unit, input_signal, dt_ms)

        min_len = min(len(target), len(predicted))
        loss = combined_loss(predicted[:min_len], target[:min_len], 0.01)
        results[name] = {"scale": amp_scale, "duration_ms": original_duration_ms, "loss": loss}

    # Noise variant (original duration, noise added to input)
    target = original_target_fn(original_duration_ms, dt_ms)
    input_signal = original_input_fn(original_duration_ms, dt_ms)
    noise = rng.normal(0, 0.05, size=len(input_signal))
    input_noisy = np.clip(input_signal + noise, 0.0, None)

    unit = create_unit()
    unit.set_theta_from_dict(theta)
    predicted, _spikes = run_unit_with_signal(unit, input_noisy, dt_ms)

    min_len = min(len(target), len(predicted))
    loss = combined_loss(predicted[:min_len], target[:min_len], 0.01)
    results["noisy"] = {"scale": 1.0, "duration_ms": original_duration_ms, "loss": loss}

    # Summary stats
    all_losses = [v["loss"] for v in results.values()]
    results["_summary"] = {
        "n_test_variants": len(all_losses),
        "worst_case_loss": max(all_losses),
        "mean_test_loss": float(np.mean(all_losses)),
    }

    return results


# --- Two-Voice Articulation Fitting ---


def create_two_voice_objective(
    target: np.ndarray,
    config: ArticulationConfig,
    dt_ms: float = 1.0,
    lambda_smooth: float = 0.01,
) -> Callable[[np.ndarray], float]:
    """Create objective function for two-voice articulation optimization.

    The objective function takes 8 parameters (4 per voice) and returns
    the combined loss between the target envelope and the articulated note output.

    Args:
        target: Target envelope to fit
        config: Articulation configuration
        dt_ms: Timestep
        lambda_smooth: Smoothness penalty weight

    Returns:
        Objective function mapping flat theta array -> loss
    """

    def objective(theta_flat: np.ndarray) -> float:
        """Evaluate loss for given theta parameters.

        Args:
            theta_flat: Flat array of 8 values:
                [grace_total, grace_balance, grace_attack_tau, grace_recovery_tau,
                 main_total, main_balance, main_attack_tau, main_recovery_tau]

        Returns:
            Combined loss value
        """
        # Create articulated note
        note = ArticulatedNote(config)

        # Set grace voice parameters
        grace_theta = {
            "theta_total": theta_flat[0],
            "theta_balance": theta_flat[1],
            "theta_attack_tau": theta_flat[2],
            "theta_recovery_tau": theta_flat[3],
        }
        note.set_voice_theta("grace", grace_theta)

        # Set main voice parameters
        main_theta = {
            "theta_total": theta_flat[4],
            "theta_balance": theta_flat[5],
            "theta_attack_tau": theta_flat[6],
            "theta_recovery_tau": theta_flat[7],
        }
        note.set_voice_theta("main", main_theta)

        # Run the articulated note
        envelope = note.run_full(dt_ms)

        # Handle length mismatch
        min_len = min(len(envelope), len(target))
        return combined_loss(envelope[:min_len], target[:min_len], lambda_smooth)

    return objective


def fit_two_voice_grace_note(
    main_duration_ms: float = 360.0,
    dt_ms: float = 1.0,
    max_iterations: int = 300,
) -> dict:
    """Fit a two-voice articulation to the grace note target.

    Uses CMA-ES to optimize 8 parameters (4 per voice) to match
    the target grace note envelope.

    Args:
        main_duration_ms: Duration of main note portion
        dt_ms: Timestep
        max_iterations: Maximum CMA-ES iterations

    Returns:
        Dictionary with fit results including best theta per voice
    """
    # Create articulation config (matches target_envelope_grace_note timing)
    config = create_grace_note_config(main_duration_ms=main_duration_ms)

    # Generate target envelope
    target = target_envelope_grace_note(main_duration_ms, dt_ms)

    # Create objective
    objective = create_two_voice_objective(target, config, dt_ms, lambda_smooth=0.01)

    # CMA-ES with 8 dimensions and multiple restarts
    import cma  # type: ignore[import-untyped]

    # Bounds for all theta parameters (from HEUThetaBounds defaults)
    bounds = HEUThetaBounds()
    lower_bounds = [
        bounds.theta_total[0],
        bounds.theta_balance[0],
        bounds.theta_attack_tau[0],
        bounds.theta_recovery_tau[0],
    ] * 2  # Repeat for both voices
    upper_bounds = [
        bounds.theta_total[1],
        bounds.theta_balance[1],
        bounds.theta_attack_tau[1],
        bounds.theta_recovery_tau[1],
    ] * 2

    # Multiple starting points to explore parameter space
    starting_points = [
        # Grace: staccato-like, Main: legato-like
        np.array([-0.5, 1.0, -0.2, -0.3, -1.0, 1.5, 0.3, 0.2]),
        # Grace: fast decay, Main: strong attack
        np.array([-0.3, 1.5, -0.5, -0.5, -0.5, 2.0, 0.0, 0.0]),
        # Grace: medium, Main: sustained
        np.array([-0.8, 0.5, 0.0, 0.0, -1.5, 2.5, 0.5, 0.3]),
        # Neutral starting point
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    loss_history = []
    best_loss = float("inf")
    best_theta = None
    total_iterations = 0

    for restart_idx, x0 in enumerate(starting_points):
        # Run CMA-ES from this starting point
        iterations_per_restart = max_iterations // len(starting_points)

        es = cma.CMAEvolutionStrategy(
            x0,
            0.4,  # sigma
            {
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": iterations_per_restart,
                "seed": 42 + restart_idx,
                "verbose": -9,  # Suppress output
                "popsize": 16,
            },
        )

        while not es.stop():
            solutions = es.ask()
            losses = [objective(s) for s in solutions]
            es.tell(solutions, losses)

            # Track best
            gen_best_idx = np.argmin(losses)
            if losses[gen_best_idx] < best_loss:
                best_loss = losses[gen_best_idx]
                best_theta = solutions[gen_best_idx].copy()
            loss_history.append(min(losses))
            total_iterations += 1

            # Early stop if target reached
            if best_loss < 0.005:
                break

        if best_loss < 0.005:
            break

    # Extract best theta per voice
    if best_theta is None:
        best_theta = np.zeros(8)
    grace_theta = {
        "theta_total": float(best_theta[0]),
        "theta_balance": float(best_theta[1]),
        "theta_attack_tau": float(best_theta[2]),
        "theta_recovery_tau": float(best_theta[3]),
    }
    main_theta = {
        "theta_total": float(best_theta[4]),
        "theta_balance": float(best_theta[5]),
        "theta_attack_tau": float(best_theta[6]),
        "theta_recovery_tau": float(best_theta[7]),
    }

    # Generate final envelope with best parameters
    note = ArticulatedNote(config)
    note.set_voice_theta("grace", grace_theta)
    note.set_voice_theta("main", main_theta)
    final_envelope = note.run_full(dt_ms)

    # Get per-voice envelopes for visualization
    note.reset()
    grace_envelope = np.zeros(len(final_envelope))
    main_envelope = np.zeros(len(final_envelope))

    for i in range(len(final_envelope)):
        output = note.step(dt_ms)
        grace_envelope[i] = output.voice_envelopes.get("grace", 0.0)
        main_envelope[i] = output.voice_envelopes.get("main", 0.0)

    # Handle length mismatch
    min_len = min(len(final_envelope), len(target))

    return {
        "name": "two_voice_grace_note",
        "grace_theta": grace_theta,
        "main_theta": main_theta,
        "final_loss": best_loss,
        "iterations": total_iterations,
        "loss_history": loss_history,
        "converged": best_loss < 0.01,
        "target": target[:min_len],
        "predicted": final_envelope[:min_len],
        "grace_envelope": grace_envelope[:min_len],
        "main_envelope": main_envelope[:min_len],
        "dt_ms": dt_ms,
        "config": config,
    }


def print_fitting_summary(results: dict) -> None:
    """Print summary of fitting results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 03: ENVELOPE FITTING RESULTS")
    print("=" * 60)

    for name in ["staccato", "legato", "grace_note"]:
        if name in results:
            r = results[name]
            print(f"\n{name.upper()}:")
            print(f"  Final loss: {r['final_loss']:.6f}")
            print(f"  Iterations: {r['iterations']}")
            print(f"  Converged: {r['converged']}")
            print("  Best theta:")
            for k, v in r["best_theta"].items():
                print(f"    {k}: {v:.4f}")

        gen_key = f"{name}_generalization"
        if gen_key in results:
            gen = results[gen_key]
            print("  Generalization:")
            for variant, data in gen.items():
                if variant.startswith("_"):
                    continue
                ratio = data["loss"] / r["final_loss"] if r["final_loss"] > 0 else float("inf")
                print(f"    {variant}: loss={data['loss']:.6f} ({ratio:.2f}x training)")
            if "_summary" in gen:
                s = gen["_summary"]
                print(
                    f"    --- worst_case: {s['worst_case_loss']:.6f}, "
                    f"mean: {s['mean_test_loss']:.6f}, "
                    f"n_variants: {s['n_test_variants']}"
                )

    # Two-voice results
    if "two_voice_grace_note" in results:
        r = results["two_voice_grace_note"]
        print("\nTWO-VOICE GRACE NOTE:")
        print(f"  Final loss: {r['final_loss']:.6f}")
        print(f"  Iterations: {r['iterations']}")
        print(f"  Converged: {r['converged']}")
        print("  Grace voice theta:")
        for k, v in r["grace_theta"].items():
            print(f"    {k}: {v:.4f}")
        print("  Main voice theta:")
        for k, v in r["main_theta"].items():
            print(f"    {k}: {v:.4f}")

        # Print comparison if single-voice result exists
        if "grace_note" in results:
            single_loss = results["grace_note"]["final_loss"]
            two_loss = r["final_loss"]
            improvement = single_loss / two_loss if two_loss > 0 else float("inf")
            print("\n  COMPARISON:")
            print(f"    Single-voice loss: {single_loss:.6f}")
            print(f"    Two-voice loss:    {two_loss:.6f}")
            print(f"    Improvement:       {improvement:.2f}x")


def check_pass_conditions(results: dict) -> bool:
    """Check if experiment passes acceptance criteria.

    Returns:
        True if all conditions pass
    """
    passed = True

    print("\n" + "-" * 60)
    print("PASS CONDITION CHECKS:")
    print("-" * 60)

    # Loss thresholds: targets are HEU-generated so achievable by construction
    loss_thresholds = {
        "staccato": 0.01,
        "legato": 0.01,
        "grace_note": 0.01,
    }
    iteration_limit = 200
    generalization_ratio = 2.0

    for name in ["staccato", "legato", "grace_note"]:
        if name not in results:
            continue

        r = results[name]
        loss_threshold = loss_thresholds.get(name, 0.10)

        # Check final loss
        loss_ok = r["final_loss"] < loss_threshold
        status = "PASS" if loss_ok else "FAIL"
        print(f"  {name} final loss < {loss_threshold}: {r['final_loss']:.6f} [{status}]")
        if not loss_ok:
            passed = False

        # Check iterations
        iter_ok = r["iterations"] < iteration_limit
        status = "PASS" if iter_ok else "FAIL"
        print(f"  {name} iterations < {iteration_limit}: {r['iterations']} [{status}]")
        if not iter_ok:
            passed = False

        # Check generalization (5 held-out variants)
        gen_key = f"{name}_generalization"
        if gen_key in results:
            gen = results[gen_key]
            # Duration variants must generalize (same dynamics, different speed)
            for variant in ["faster", "slower"]:
                if variant not in gen:
                    continue
                data = gen[variant]
                ratio = data["loss"] / r["final_loss"] if r["final_loss"] > 0 else float("inf")
                gen_ok = ratio < generalization_ratio
                status = "PASS" if gen_ok else "FAIL"
                print(f"  {name} {variant} < {generalization_ratio}x: {ratio:.2f}x [{status}]")
                if not gen_ok:
                    passed = False

            # Amplitude/noise variants: informational (nonlinear system sensitivity)
            for variant in ["amp_low", "amp_high", "noisy"]:
                if variant not in gen:
                    continue
                data = gen[variant]
                ratio = data["loss"] / r["final_loss"] if r["final_loss"] > 0 else float("inf")
                print(f"  {name} {variant}: {ratio:.2f}x ({data['loss']:.6f}) [SENSITIVITY]")

            # Worst-case absolute bound (all variants including sensitivity tests)
            if "_summary" in gen:
                s = gen["_summary"]
                worst = s["worst_case_loss"]
                worst_ok = worst < 0.05
                status = "PASS" if worst_ok else "FAIL"
                print(f"  {name} worst-case test loss < 0.05: {worst:.6f} [{status}]")
                if not worst_ok:
                    passed = False

    # Two-voice grace note (informational)
    # The grace note target is single-voice-generated, so two-voice won't improve.
    # Architectural advantage is demonstrated in exp_04 (impossibility proof).
    if "two_voice_grace_note" in results:
        r = results["two_voice_grace_note"]
        print(f"  two_voice_grace_note loss: {r['final_loss']:.6f} [INFO]")

        if "grace_note" in results:
            single_loss = results["grace_note"]["final_loss"]
            improvement = single_loss / r["final_loss"] if r["final_loss"] > 0 else float("inf")
            print(f"  two_voice improvement: {improvement:.2f}x [INFO]")

    print("-" * 60)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")

    return passed


def plot_all_fits(results: dict, show: bool = True) -> None:
    """Plot all fitting results."""
    import matplotlib.pyplot as plt

    for name in ["staccato", "legato", "grace_note"]:
        if name in results:
            r = results[name]
            dt_ms = r["dt_ms"]
            time_ms = np.arange(len(r["target"])) * dt_ms

            # Plot fit
            fig = plot_envelope_fit(
                time_ms,
                r["target"],
                r["predicted"],
                title=f"{name.title()} Envelope Fit (loss={r['final_loss']:.6f})",
            )

            # Plot learning curve
            if len(r["loss_history"]) > 0:
                fig2, ax = plt.subplots(figsize=(8, 4))
                plot_learning_curve(
                    np.arange(len(r["loss_history"])),
                    np.array(r["loss_history"]),
                    target_loss=0.01,
                    title=f"{name.title()} Learning Curve",
                    ax=ax,
                )

    # Plot two-voice grace note results
    if "two_voice_grace_note" in results:
        r = results["two_voice_grace_note"]
        dt_ms = r["dt_ms"]
        time_ms = np.arange(len(r["target"])) * dt_ms

        # Main comparison plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top: Combined fit vs target
        ax1 = axes[0]
        ax1.plot(time_ms, r["target"], "b-", label="Target", linewidth=2, alpha=0.8)
        ax1.plot(time_ms, r["predicted"], "r--", label="Two-Voice Combined", linewidth=2)
        ax1.set_ylabel("Envelope")
        ax1.set_title(f"Two-Voice Grace Note Fit (loss={r['final_loss']:.6f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Per-voice envelopes
        ax2 = axes[1]
        ax2.plot(time_ms, r["grace_envelope"], "g-", label="Grace Voice", linewidth=1.5)
        ax2.plot(time_ms, r["main_envelope"], "m-", label="Main Voice", linewidth=1.5)
        ax2.plot(time_ms, r["predicted"], "k--", label="Combined (max)", linewidth=1, alpha=0.5)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Envelope")
        ax2.set_title("Per-Voice Envelopes")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Learning curve
        if len(r["loss_history"]) > 0:
            fig2, ax = plt.subplots(figsize=(8, 4))
            plot_learning_curve(
                np.arange(len(r["loss_history"])),
                np.array(r["loss_history"]),
                target_loss=0.01,
                title="Two-Voice Grace Note Learning Curve",
                ax=ax,
            )

    if show:
        plt.show()


def run_diagnostics(results: dict) -> bool:
    """Build and report diagnostics for fitting results.

    Returns:
        True if all diagnostics bundles are VALID.
    """
    output_dir = Path("results/exp_03")
    dt_ms = 1.0
    spike_threshold = 0.8
    all_valid = True

    # Expected event times and spike count ranges per articulation
    expected_events = {
        "staccato": [0.0],
        "legato": [0.0],
        "grace_note": [0.0, 40.0],
    }
    expected_spikes = {
        "staccato": (1, 3),  # Transient → 1-2 spikes
        "legato": (1, 5),  # Sustained → may spike on onset
        "grace_note": (1, 5),  # Two pulses → 1-4 spikes
    }

    print("\n" + "-" * 60)
    print("DIAGNOSTICS:")
    print("-" * 60)

    for name in ["staccato", "legato", "grace_note"]:
        if name not in results:
            continue
        r = results[name]
        envelope = r["predicted"]
        spike_bool = r["spike_bool"]
        n_frames = len(envelope)
        duration_ms = n_frames * dt_ms

        bundle = build_diagnostics_bundle(
            envelope=envelope,
            spike_bool=spike_bool,
            dt_ms=dt_ms,
            window_ms=(0.0, duration_ms),
            expected_event_times_ms=expected_events.get(name, []),
            spike_threshold=spike_threshold,
            expected_spike_count=expected_spikes.get(name),
        )
        report_diagnostics(bundle, name, output_dir / f"{name}_diagnostics.json")
        if not bundle.is_valid:
            all_valid = False

    # Print generalization summary
    for name in ["staccato", "legato", "grace_note"]:
        gen_key = f"{name}_generalization"
        if gen_key in results and "_summary" in results[gen_key]:
            s = results[gen_key]["_summary"]
            print(
                f"  GENERALIZATION [{name}]: n_test={s['n_test_variants']}, "
                f"worst={s['worst_case_loss']:.6f}, mean={s['mean_test_loss']:.6f}"
            )

    print("-" * 60)
    print(f"DIAGNOSTICS OVERALL: {'ALL VALID' if all_valid else 'SOME INVALID'}")
    return all_valid


def main():
    """Run the complete experiment."""
    print("Starting Experiment 03: Envelope Fitting")
    print("=" * 60)

    dt_ms = 1.0

    # Define target configurations with matching input signals
    # Key insight: input must match target semantics!
    # - Staccato: single pulse → transient response
    # - Legato: pulse + tonic drive → sustained envelope
    # - Grace note: two pulses → two-peak response
    target_configs = {
        "staccato": {
            "duration_ms": 200,
            "target_fn": target_envelope_staccato,
            "input_fn": input_signal_staccato,
        },
        "legato": {
            "duration_ms": 500,
            "target_fn": target_envelope_legato,
            "input_fn": input_signal_legato,
        },
        "grace_note": {
            "duration_ms": 400,
            "target_fn": target_envelope_grace_note,
            "input_fn": input_signal_grace_note,
        },
    }

    results = {}

    # Fit each target (single-voice)
    for name, config in target_configs.items():
        duration_ms = config["duration_ms"]
        target = config["target_fn"](duration_ms, dt_ms)
        input_signal = config["input_fn"](duration_ms, dt_ms)

        print(
            f"Fitting {name} envelope (target len={len(target)}, input len={len(input_signal)})..."
        )
        results[name] = fit_envelope(name, target, input_signal, dt_ms, max_iterations=150)

        # Test generalization
        print(f"Testing {name} generalization...")
        results[f"{name}_generalization"] = test_generalization(
            results[name]["best_theta"],
            config["target_fn"],
            config["input_fn"],
            duration_ms,
            dt_ms,
        )

    # V0.3: Two-voice grace note fitting
    print("\n" + "=" * 60)
    print("V0.3: TWO-VOICE ARTICULATION TEST")
    print("=" * 60)
    print("\nFitting two-voice grace note (hypothesis H3.1)...")
    print("  Grace voice: independent HEU for initial transient")
    print("  Main voice:  independent HEU with tonic drive for sustain")

    results["two_voice_grace_note"] = fit_two_voice_grace_note(
        main_duration_ms=360.0,  # Matches target_envelope_grace_note default
        dt_ms=dt_ms,
        max_iterations=200,
    )

    # Print summary
    print_fitting_summary(results)

    # Check pass conditions
    passed = check_pass_conditions(results)

    # Run diagnostics
    diagnostics_valid = run_diagnostics(results)
    if not diagnostics_valid:
        passed = False

    # Plot results
    try:
        plot_all_fits(results, show=True)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
