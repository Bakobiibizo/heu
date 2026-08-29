"""
Detailed diagnostic of the main attack region and grace peak issues.
"""

import numpy as np

from hh_envelope.polyphony.articulation import ArticulatedNote, create_grace_note_config
from hh_envelope.utils.signals import target_envelope_grace_note


def analyze_attack_transition():
    """Analyze the critical transition from grace to main."""
    grace_theta = {
        "theta_total": 0.2151,
        "theta_balance": -1.9128,
        "theta_attack_tau": -0.9876,
        "theta_recovery_tau": 0.5507,
    }
    main_theta = {
        "theta_total": -0.3215,
        "theta_balance": 2.9999,
        "theta_attack_tau": 0.9992,
        "theta_recovery_tau": -0.2093,
    }

    config = create_grace_note_config(main_duration_ms=360.0)
    note = ArticulatedNote(config)
    note.set_voice_theta("grace", grace_theta)
    note.set_voice_theta("main", main_theta)

    target = target_envelope_grace_note(360.0, 1.0)

    # Run and collect detailed data
    note.reset()
    num_samples = int(config.total_duration_ms)

    data = []
    for i in range(num_samples):
        output = note.step(1.0)
        grace = output.voice_envelopes.get("grace", 0.0)
        main = output.voice_envelopes.get("main", 0.0)
        combined = output.combined_envelope
        target_val = target[i] if i < len(target) else 0.0

        data.append(
            {
                "time_ms": i,
                "grace": grace,
                "main": main,
                "combined": combined,
                "target": target_val,
                "error": abs(combined - target_val),
            }
        )

    # Print detailed analysis of critical regions
    print("=" * 80)
    print("DETAILED TRANSITION ANALYSIS")
    print("=" * 80)

    # Grace peak region (0-10ms)
    print("\n1. GRACE PEAK REGION (0-10ms)")
    print("-" * 60)
    print(f"{'Time':>6} {'Target':>8} {'Grace':>8} {'Main':>8} {'Combined':>8} {'Error':>8}")
    for d in data[:11]:
        print(
            f"{d['time_ms']:>6} {d['target']:>8.4f} {d['grace']:>8.4f} {d['main']:>8.4f} {d['combined']:>8.4f} {d['error']:>8.4f}"
        )

    # Gap region (30-40ms)
    print("\n2. GAP REGION (28-42ms)")
    print("-" * 60)
    print(f"{'Time':>6} {'Target':>8} {'Grace':>8} {'Main':>8} {'Combined':>8} {'Error':>8}")
    for d in data[28:43]:
        print(
            f"{d['time_ms']:>6} {d['target']:>8.4f} {d['grace']:>8.4f} {d['main']:>8.4f} {d['combined']:>8.4f} {d['error']:>8.4f}"
        )

    # Main attack region (40-70ms)
    print("\n3. MAIN ATTACK REGION (40-70ms)")
    print("-" * 60)
    print(f"{'Time':>6} {'Target':>8} {'Grace':>8} {'Main':>8} {'Combined':>8} {'Error':>8}")
    for d in data[40:71]:
        print(
            f"{d['time_ms']:>6} {d['target']:>8.4f} {d['grace']:>8.4f} {d['main']:>8.4f} {d['combined']:>8.4f} {d['error']:>8.4f}"
        )

    # Calculate statistics
    grace_region = data[:30]
    gap_region = data[30:40]
    main_attack = data[40:70]
    main_sustain = data[70:]

    print("\n4. REGIONAL ERROR STATISTICS")
    print("-" * 60)
    print(
        f"   Grace (0-30ms):     max_error={max(d['error'] for d in grace_region):.4f}, mean={np.mean([d['error'] for d in grace_region]):.4f}"
    )
    print(
        f"   Gap (30-40ms):      max_error={max(d['error'] for d in gap_region):.4f}, mean={np.mean([d['error'] for d in gap_region]):.4f}"
    )
    print(
        f"   Main attack (40-70ms): max_error={max(d['error'] for d in main_attack):.4f}, mean={np.mean([d['error'] for d in main_attack]):.4f}"
    )
    print(
        f"   Main sustain (70+ms): max_error={max(d['error'] for d in main_sustain):.4f}, mean={np.mean([d['error'] for d in main_sustain]):.4f}"
    )

    # Identify worst errors
    print("\n5. TOP 10 WORST ERRORS")
    print("-" * 60)
    sorted_data = sorted(data, key=lambda d: d["error"], reverse=True)[:10]
    print(f"{'Time':>6} {'Target':>8} {'Combined':>8} {'Error':>8}")
    for d in sorted_data:
        print(f"{d['time_ms']:>6} {d['target']:>8.4f} {d['combined']:>8.4f} {d['error']:>8.4f}")


def analyze_grace_peak_constraint():
    """Why can't we achieve a 0.6 grace peak?"""
    print("\n" + "=" * 80)
    print("GRACE PEAK CONSTRAINT ANALYSIS")
    print("=" * 80)

    # The target grace peak is 0.6
    # With pulse amplitude 0.7, we're getting 0.46
    # This suggests the HEU is not amplifying enough, or wrong parameters

    from hh_envelope.core import HHEnvelopeUnit

    print("\n1. EFFECT OF PULSE AMPLITUDE ON PEAK")
    print("-" * 60)
    print(f"{'Pulse Amp':>10} {'Peak':>8} {'Peak Time':>10}")

    for pulse_amp in [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
        unit = HHEnvelopeUnit()
        unit.set_theta_from_dict(
            {
                "theta_total": 0.2151,
                "theta_balance": -1.9128,
                "theta_attack_tau": -0.9876,
                "theta_recovery_tau": 0.5507,
            }
        )
        unit.reset()

        envelope = []
        for i in range(100):
            excitation = pulse_amp if i == 0 else 0.0
            output = unit.step(excitation, dt_ms=1.0)
            envelope.append(output.envelope)

        peak = max(envelope)
        peak_time = envelope.index(peak)
        print(f"{pulse_amp:>10.1f} {peak:>8.4f} {peak_time:>10}ms")

    print("\n2. EFFECT OF THETA PARAMETERS ON PEAK (pulse=0.7)")
    print("-" * 60)

    # Try different theta_balance values
    print("\nVarying theta_balance (attack/recovery ratio):")
    print(f"{'theta_balance':>14} {'Peak':>8}")
    for balance in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        unit = HHEnvelopeUnit()
        unit.set_theta_from_dict(
            {
                "theta_total": 0.2151,
                "theta_balance": balance,
                "theta_attack_tau": -0.9876,
                "theta_recovery_tau": 0.5507,
            }
        )
        unit.reset()

        envelope = []
        for i in range(100):
            excitation = 0.7 if i == 0 else 0.0
            output = unit.step(excitation, dt_ms=1.0)
            envelope.append(output.envelope)

        peak = max(envelope)
        print(f"{balance:>14.1f} {peak:>8.4f}")

    # Try different theta_total values
    print("\nVarying theta_total (total conductance):")
    print(f"{'theta_total':>12} {'Peak':>8}")
    for total in [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]:
        unit = HHEnvelopeUnit()
        unit.set_theta_from_dict(
            {
                "theta_total": total,
                "theta_balance": -1.9128,
                "theta_attack_tau": -0.9876,
                "theta_recovery_tau": 0.5507,
            }
        )
        unit.reset()

        envelope = []
        for i in range(100):
            excitation = 0.7 if i == 0 else 0.0
            output = unit.step(excitation, dt_ms=1.0)
            envelope.append(output.envelope)

        peak = max(envelope)
        print(f"{total:>12.1f} {peak:>8.4f}")


def main():
    analyze_attack_transition()
    analyze_grace_peak_constraint()

    print("\n" + "=" * 80)
    print("SUMMARY: ROOT CAUSES OF PERFORMANCE GAP")
    print("=" * 80)
    print("""
1. GRACE PEAK CONSTRAINT (Target=0.60, Got=0.46)
   - The grace voice with pulse amplitude 0.7 cannot reach 0.6 peak
   - The optimized theta_balance=-1.91 (recovery-biased) limits peak
   - Increasing pulse amplitude or theta_total could help but affects decay

2. MAIN ATTACK DELAY (Highest per-sample loss at 40-70ms)
   - Target rises from 0.15 to 1.0 in ~15ms
   - Two-voice rises more slowly due to:
     a) Main voice theta_attack_tau=1.0 at upper bound (slow attack)
     b) Optimizer constrained by bounds
   - This creates a timing mismatch during the critical transition

3. PARAMETER BOUND CONSTRAINTS
   - 3 of 8 parameters are at their bounds:
     * grace.theta_attack_tau = -0.99 (at -1.0 lower bound)
     * main.theta_balance = 3.0 (at 3.0 upper bound)
     * main.theta_attack_tau = 1.0 (at 1.0 upper bound)
   - Optimizer wants to go further but can't

4. MAIN SUSTAIN REGION DOMINATES
   - Main sustain is 82.5% of total samples
   - Even small per-sample error accumulates
   - Target sustain ~0.84, we produce ~0.95-1.0

POTENTIAL SOLUTIONS:
   1. Increase theta_attack_tau bounds to [-2.0, 2.0]
   2. Increase grace pulse amplitude to 1.0+ for higher peak
   3. Use tonic drive on grace voice to control decay more precisely
   4. Consider time-varying parameters within articulation
""")


if __name__ == "__main__":
    main()
