# HEU Experiment Report

Full reproduction run, 2026-02-02. All experiments executed with `uv run python -m hh_envelope.experiments.<name>` on Python 3.14.2.

---

## Unit Tests

**38/38 PASS** (0.11s)

| Suite | Tests | Status |
|-------|-------|--------|
| TestHEUState | 6 | PASS |
| TestBackwardsCompatibility | 2 | PASS |
| TestHHEnvelopeUnitBasic | 4 | PASS |
| TestPulseResponse | 2 | PASS |
| TestRefractory | 2 | PASS |
| TestTonicSustain | 2 | PASS |
| TestDeterminism | 2 | PASS |
| TestSafetyDamping | 2 | PASS |
| TestThetaParameters | 3 | PASS |
| TestSerialization | 2 | PASS |
| TestModulators | 1 | PASS |
| TestLookahead | 5 | PASS |
| TestOversampling | 5 | PASS |

---

## Experiment 01: Tempo Locking

**Status: PASS**

### Hypothesis

A single HEU driven by a pulse train will phase-lock to the driving tempo, exhibit stable refractory spacing, and show asymmetry in response to accented pulses.

### Results

| Tempo | PLV | Mean ISI (ms) | ISI CV | Min ISI (ms) | Peak CV |
|-------|-----|---------------|--------|--------------|---------|
| 90 BPM | 1.000 | 666.7 | 0.001 | 666.0 | 0.025 |
| 120 BPM | 1.000 | 500.0 | 0.000 | 500.0 | 0.022 |
| 140 BPM | 1.000 | 428.6 | 0.001 | 428.0 | 0.020 |

**Accent test:** Ratio = 1.19 (input strength 1.5x). The HEU compresses dynamic range — accents are differentiated without saturation.

**Jitter test (15ms Gaussian):** PLV = 0.987. Phase locking degrades gracefully under humanized timing.

### Pass Conditions

| Condition | Required | Actual | Status |
|-----------|----------|--------|--------|
| PLV >= 0.8 (all tempos) | 0.8 | 1.000 | PASS |
| Min ISI >= refractory (20ms) | 20 | 428+ | PASS |
| Accent ratio in [1.1, 2.0] | 1.1–2.0 | 1.19 | PASS |

### Diagnostics

All 5 diagnostic bundles VALID (tempo_90, tempo_120, tempo_140, accents, jitter).

### Interpretation

The HEU achieves perfect phase locking at all tested tempos. ISI coefficient of variation is near-zero, confirming deterministic spike timing. The refractory mechanism prevents double-firing (minimum ISI far exceeds refractory period). Accent sensitivity is preserved with compression — the 1.5x input maps to 1.19x peak, consistent with the sigmoid nonlinearity in the attack gate.

---

## Experiment 02: Two-Voice Coupling

**Status: PASS (stability) / INCONCLUSIVE (coupling effect)**

### Hypothesis

Two HEUs with weak coupling and interval bias will maintain stable relative phase under shared pulse drive, exhibit consonance-biased co-activation, and avoid chaotic runaway.

### Results

All 9 conditions (3 intervals x 3 coupling strengths) produce identical results:

| Interval | Semitones | Weight | Correlation | Phase Coherence | Spikes (each) | Activity EMA |
|----------|-----------|--------|-------------|-----------------|---------------|-------------|
| Unison | 0 | +1.00 | 1.000 | 1.000 | 19 | 0.303 |
| Fifth | 7 | +0.80 | 1.000 | 1.000 | 19 | 0.303 |
| Tritone | 6 | -0.20 | 1.000 | 1.000 | 19 | 0.303 |

Coupling strengths tested: 0.05, 0.1, 0.2. All produce identical behavior.

### Pass Conditions

| Condition | Required | Actual | Status |
|-----------|----------|--------|--------|
| Consonance ordering | unison >= fifth >= tritone | All equal (1.0) | PASS (vacuously) |
| Stability (activity < ceiling) | < 0.8 | 0.303 | PASS |
| All voices responsive | spikes > 0 | 19 each | PASS |

### Diagnostic Invariant Failures

**9/9 coupling effect invariants FAIL.** Mean envelope difference = 0.0, peak difference = 0.0 across all conditions.

### Interpretation

The coupling mechanism is **stable but inert**. The 500ms pulse spacing (120 BPM) causes each voice's envelope to fully decay before the next pulse arrives. At pulse time, the partner's envelope is approximately zero, so the coupling term $c_i = \text{strength} \cdot w_{ij} \cdot v^{(j)}(t) \approx 0$ regardless of strength or interval weight.

This validates that coupling does not break the system. It does not validate that coupling produces meaningful inter-voice coordination. A revised protocol with overlapping envelopes (faster tempo, sustained tonic, staggered onsets) is required.

**Paper treatment:** This experiment is placed in Appendix A.11 as supplementary material, with explicit acknowledgment that coupling effectiveness is untested.

---

## Experiment 03: Envelope Fitting

**Status: PASS**

### Hypothesis

CMA-ES optimization can fit the HEU to target envelope shapes for staccato, legato, and grace note articulations, converging within 200 iterations to loss < 0.01.

### Results

| Articulation | Loss | Iterations | Converged |
|-------------|------|------------|-----------|
| Staccato | 0.000125 | 1 | Yes |
| Legato | 0.000578 | 4 | Yes |
| Grace note | 0.000345 | 4 | Yes |

All losses are 17-80x below the 0.01 threshold.

### Optimized Parameters

| Parameter | Staccato | Legato | Grace Note |
|-----------|----------|--------|------------|
| theta_total | -0.274 | -1.442 | -0.956 |
| theta_balance | +0.462 | +2.471 | +1.165 |
| theta_attack_tau | -0.776 | -0.964 | -1.344 |
| theta_recovery_tau | +1.232 | -0.701 | +0.738 |

Note: The optimizer finds different theta than the reference values while achieving near-zero loss, demonstrating a well-conditioned parameter landscape with multiple equivalent solutions.

### Generalization (tempo scaling)

| Articulation | Faster (0.75x) | Slower (1.25x) |
|-------------|----------------|----------------|
| Staccato | 1.25x training | 0.83x training |
| Legato | 1.24x training | 0.84x training |
| Grace note | 0.87x training | 0.95x training |

All tempo-scaled generalizations remain well within bounds.

### Generalization (amplitude and noise)

| Articulation | Amp Low | Amp High | Noisy | Worst Case |
|-------------|---------|----------|-------|------------|
| Staccato | 12.7x | 1.9x | 39.7x | 0.005 |
| Legato | 22.7x | 15.2x | 3.5x | 0.013 |
| Grace note | 30.1x | 14.3x | 10.0x | 0.010 |

Amplitude perturbation shows higher sensitivity than tempo scaling, consistent with the nonlinear sigmoid dynamics. All worst-case losses remain below 0.05.

### Two-Voice Grace Note (INFO only)

| Metric | Value |
|--------|-------|
| Two-voice loss | 0.004612 |
| Single-voice loss | 0.000345 |
| Comparison | 0.07x (single-voice target is easier) |

The single-voice grace note target is a simpler shape than the two-voice impossibility target (Exp 04). The two-voice architecture is not needed for this target.

### Diagnostics

All 3 diagnostic bundles VALID (staccato, legato, grace_note).

### Interpretation

The optimizer navigates the HEU parameter landscape effectively. CMA-ES finds near-optimal solutions in 1-4 iterations across all articulations. The parameter space is well-conditioned: different theta configurations produce equivalent envelopes, indicating a smooth loss surface without pathological local minima.

This result is a necessary precondition for Experiment 04. If the optimizer could not reliably find good solutions for achievable targets, poor performance on the impossibility target could be attributed to optimization failure rather than architectural limitation.

---

## Experiment 04: Impossibility Proof (Single-State Separation)

**Status: PASS**

### Hypothesis (H3.2)

A target constructed from two-voice HEU dynamics will exhibit:
- Single-voice loss > 0.005 (constrained by parameter conflicts)
- Two-voice loss < 0.005 (achievable by construction)
- Improvement > 3x (meaningful architectural advantage)
- Single-voice parameters at bounds >= 2/4

### Results

| Architecture | Loss | Params at Bounds |
|-------------|------|------------------|
| Single-voice ($\mathcal{H}_{1c}$) | 0.0196 | 2/4 (50%) |
| Two-voice ($\mathcal{H}_2$) | 0.000927 | 0/8 (0%) |
| **Improvement** | **21.18x** | |

### Single-Voice Optimized Parameters

| Parameter | Value | Bound | Status |
|-----------|-------|-------|--------|
| theta_total | -1.470 | [-2, +2] | within bounds |
| theta_balance | +3.000 | [-3, +3] | **AT UPPER BOUND** |
| theta_attack_tau | +0.273 | [-2, +2] | within bounds |
| theta_recovery_tau | -2.000 | [-2, +2] | **AT LOWER BOUND** |

The optimizer pushes theta_balance to maximum (attack-biased, for sustain) and theta_recovery_tau to minimum (fastest recovery, for transient decay). These are conflicting objectives.

### Two-Voice Optimized Parameters

| Parameter | Grace Voice | Main Voice |
|-----------|-------------|------------|
| theta_total | -0.328 | -0.839 |
| theta_balance | +0.119 | +2.306 |
| theta_attack_tau | -0.338 | +0.816 |
| theta_recovery_tau | -0.493 | +0.466 |

Parameter divergence between voices on theta_balance: 2.19 units. This configuration is impossible for any single HEU.

### Regional MSE Breakdown

| Region | Single MSE | Two MSE | Improvement |
|--------|------------|---------|-------------|
| Grace (0-25ms) | 0.077 | 0.006 | 12.8x |
| Gap (25-50ms) | 0.152 | 0.00005 | **2888x** |
| Main attack (50-80ms) | 0.0002 | 0.0007 | 0.4x |
| Main sustain (80ms+) | 0.007 | 0.0006 | 11.4x |

The **gap region** is the decisive discriminator. Single-voice cannot simultaneously decay fast from the transient peak while maintaining capacity for flat sustain.

### Pass Conditions

| Condition | Required | Actual | Status |
|-----------|----------|--------|--------|
| Single-voice loss > 0.005 | > 0.005 | 0.0196 | PASS |
| Two-voice loss < 0.005 | < 0.005 | 0.000927 | PASS |
| Improvement > 3x | > 3x | 21.18x | PASS |
| Params at bounds >= 2/4 | >= 2 | 2/4 | PASS |

### Diagnostics

Both diagnostic bundles VALID (single_voice, two_voice).

### Interpretation

The optimizer's consistent convergence to boundary values (reproduced across 4 independent restarts with identical results) is the empirical signature of architectural insufficiency. The single-voice HEU cannot represent the target's conflicting temporal regimes. The two-voice solution resolves this by allowing each voice to specialize: the grace voice uses recovery-biased parameters for fast decay, while the main voice uses attack-biased parameters for flat sustain.

The 2888x gap region improvement is the fundamental capability difference. This is not an optimization artifact — both architectures use identical CMA-ES settings.

---

## Experiment 05: Uncoupled-Parameter Ablation

**Status: PASS (Outcome A — uncoupled still fails)**

### Hypothesis

Removing the balance coupling (allowing independent attack and recovery strengths) will reveal whether the expressivity separation is due to the coupling constraint or the single-state constraint.

### Three-Condition Comparison

| Architecture | Loss | Params at Bounds | vs Coupled |
|-------------|------|------------------|------------|
| Coupled ($\mathcal{H}_{1c}$) | 0.0196 | 2/4 (50%) | — |
| Uncoupled ($\mathcal{H}_{1u}$) | 0.0071 | 2/4 (50%) | 2.75x better |
| Two-voice ($\mathcal{H}_2$) | 0.0009 | 0/8 (0%) | 21.2x better |

### Improvement Ratios

| Comparison | Ratio |
|-----------|-------|
| Coupled / Two-voice | 21.18x |
| Uncoupled / Two-voice | 7.71x |
| Coupled / Uncoupled | 2.75x |

### Uncoupled Parameters

| Parameter | Value | Bound | Status |
|-----------|-------|-------|--------|
| theta_attack_strength | +1.635 | [-3, +3] | within bounds |
| theta_recovery_strength | -3.000 | [-3, +3] | **AT LOWER BOUND** |
| theta_attack_tau | -0.202 | [-2, +2] | within bounds |
| theta_recovery_tau | +2.000 | [-2, +2] | **AT UPPER BOUND** |

The uncoupled optimizer pushes recovery strength to minimum and recovery tau to maximum — both attempts to minimize recovery's effect, allowing the attack-dominated dynamics needed for sustain. But the single state still cannot decay fast enough for the transient region.

### Regional MSE Breakdown

| Region | Coupled | Uncoupled | Two-Voice | Unc/Two |
|--------|---------|-----------|-----------|---------|
| Grace (0-25ms) | 0.077 | 0.050 | 0.006 | 8.3x |
| Gap (25-50ms) | 0.152 | 0.027 | 0.00005 | **510x** |
| Main attack (50-80ms) | 0.0002 | 0.023 | 0.0007 | 35.9x |
| Main sustain (80ms+) | 0.007 | 0.001 | 0.0006 | 1.4x |

### Decomposition of the 21x Gap

$$21.2\times = 2.75\times \text{ (coupling)} \times 7.7\times \text{ (single-state)}$$

The balance coupling accounts for approximately 2.75x of the total 21x separation. The remaining 7.7x is attributable to the single-state constraint itself.

### Interpretation

**Outcome A confirmed:** The separation persists after removing the balance coupling. Within this HEU family, the primary bottleneck is the single envelope state (7.7x), not the parameter coupling (2.75x). The gap region — where the conflict between fast decay and stable sustain is most acute — remains 510x worse even with uncoupled parameters.

This establishes that the expressivity separation is architectural (number of dynamical states) rather than parametric (how those states are governed).

---

## Summary

| Experiment | Status | Key Metric | Paper Section |
|-----------|--------|------------|---------------|
| Unit Tests | **38/38 PASS** | 0.11s | — |
| Exp 01: Tempo Locking | **PASS** | PLV = 1.0, accent = 1.19 | Section 4 |
| Exp 02: Two-Voice Coupling | **PASS** (stability) | Stable but inert | Appendix A.11 |
| Exp 03: Envelope Fitting | **PASS** | Loss < 0.001 all articulations | Section 4 |
| Exp 04: Impossibility | **PASS** | 21.18x separation, 2888x gap | Section 3 |
| Exp 05: Ablation | **PASS** (Outcome A) | 7.7x uncoupled gap, 510x gap region | Section 3 |

### Consistency Check: Paper vs Reproduction

| Paper Claim | Paper Value | Reproduction Value | Match |
|------------|-------------|-------------------|-------|
| Exp 01 PLV at 120 BPM | 1.000 | 1.000 | Yes |
| Exp 01 accent ratio | 1.19 | 1.19 | Yes |
| Exp 01 jitter PLV | 0.987 | 0.987 | Yes |
| Exp 03 staccato loss | 0.000125 | 0.000125 | Yes |
| Exp 03 legato loss | 0.000578 | 0.000578 | Yes |
| Exp 03 grace note loss | 0.000345 | 0.000345 | Yes |
| Exp 04 single-voice loss | 0.0196 | 0.0196 | Yes |
| Exp 04 two-voice loss | 0.00093 | 0.000927 | Yes |
| Exp 04 improvement | 21x | 21.18x | Yes |
| Exp 04 gap improvement | 2888x | 2888x | Yes |
| Exp 05 uncoupled loss | 0.0071 | 0.0071 | Yes |
| Exp 05 coupling ratio | 2.75x | 2.75x | Yes |
| Exp 05 single-state ratio | 7.7x | 7.71x | Yes |
| Exp 05 gap region ratio | 510x | 510x | Yes |

All paper claims exactly reproduce.
