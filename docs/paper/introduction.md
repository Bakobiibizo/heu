## Introduction

Detecting salient events in continuous signals is a fundamental problem across domains: note onsets in audio, action potentials in neural recordings, anomalies in sensor streams. The task requires more than classification—it demands **event commitment**, the decision of *when* to act given accumulating but ambiguous evidence, followed by suppression of redundant responses until new evidence warrants another commitment.

Standard approaches frame event detection as frame-wise classification followed by post-hoc peak-picking or thresholding. This two-stage pipeline introduces a fundamental mismatch: the classifier optimizes per-frame accuracy while the downstream consumer needs sparse, well-timed discrete events. The result is either fragmented outputs (multiple detections per event) or missed events (over-smoothed posteriors that fail to cross threshold).

Biological systems solve this problem differently. Neurons accumulate evidence through membrane potential dynamics, commit to action potentials when threshold is crossed, and enforce refractory periods that prevent immediate re-firing. This architecture couples evidence accumulation, threshold commitment, and temporal suppression into a single dynamical system rather than a pipeline of independent stages.

We ask: **what are the expressivity limits of excitable systems whose temporal behavior is governed by a single dynamical state with globally coupled parameters?**

---

### Contribution

We introduce the **Hierarchical Excitable Unit (HEU)**, a continuous-time dynamical system inspired by Hodgkin-Huxley neural models but designed for signal processing rather than biological simulation. The term *hierarchical* reflects the intended use of HEUs as composable primitives whose parameters can be modulated by higher-level processes; in this paper we focus on the single-unit and minimal two-voice cases. The HEU maps continuous excitation to a bounded envelope state, produces discrete spike events at threshold crossings, and enforces refractory suppression—all through coupled differential equations with a small, interpretable parameter set.

Our central contribution is a **constructive impossibility result**: we prove that a single-voice HEU cannot represent certain temporal targets regardless of parameter tuning, while a minimal two-voice extension succeeds. The result is neither asymptotic nor approximate: we exhibit a concrete temporal target for which single-voice optimization consistently saturates 3 of 4 parameters at their bounds while remaining 21× worse than a minimal two-voice solution.

The impossibility arises from **parameter coupling** in the single-voice dynamics. The HEU's attack and recovery processes share a balance parameter that trades off transient response against sustained output. A target requiring both fast transient decay *and* stable sustain creates conflicting requirements that no single parameter setting can satisfy. The two-voice architecture resolves this by assigning independent parameters to each temporal regime.

This result has practical implications. We apply the HEU to polyphonic music transcription, where the task is to convert audio to MIDI note sequences. On polyphonic piano content (9-file evaluation set, 50 ms onset tolerance), a single-voice HEU bank operating at the native spectrogram frame rate (23 ms) exhibits envelope oscillation that fragments notes into 3–5× redundant detections. Stabilizing the dynamics through temporal upsampling or hysteresis thresholds—interventions that implicitly introduce finer-grained or state-augmented dynamics—yields an order-of-magnitude improvement in F1 (0.054 → 0.731).

---

### Scope and Non-Claims

The HEU is a **non-learning** system. All behavior is determined by fixed parameters and state evolution; there are no learned weights. This is intentional: we aim to characterize the expressivity of excitable dynamics themselves, not the capacity of learned approximations.

We do not claim that the HEU outperforms state-of-the-art transcription systems, which use learned models with orders of magnitude more parameters. The transcription application demonstrates that the theoretical expressivity limit manifests as a practical performance bottleneck—not that excitable units are the optimal solution to transcription.

The impossibility result is **constructive, not universal**. We exhibit one target that single-voice cannot represent; we do not characterize the full class of such targets. The result establishes that the expressivity gap exists and is non-trivial (21×), not that it is maximal.

---

### Paper Outline

**Section 2 (Model)** formally defines the single-voice HEU: envelope dynamics, spike emission, refractory mechanism, and parameterization through normalized coordinates.

**Section 3 (Problem Setup)** constructs the impossibility target—a temporal pattern with conflicting requirements—and states the expressivity separation as a proposition.

**Section 4 (Experiments)** validates the HEU through four experiments: tempo locking (basic dynamics), two-voice coupling (stability), envelope fitting (parameter optimization), and single-voice impossibility (the main result).

**Section 5 (Application)** applies the HEU to polyphonic audio transcription, demonstrating that the theoretical bottleneck manifests as note fragmentation and that stabilization yields an order-of-magnitude F1 improvement.

**Section 6 (Related Work)** positions the HEU against Hodgkin-Huxley models, spiking neural networks, onset detection methods, and expressivity results in neural architectures.

**Section 7 (Discussion)** interprets the impossibility result, addresses limitations, and suggests extensions.

**Appendix A** provides full gate dynamics equations and parameter bounds.

---

### Summary of Results

| Claim | Evidence |
|-------|----------|
| Single HEU phase-locks to rhythmic input | PLV = 1.0 at 90–140 BPM (Exp. 1) |
| Parameter optimization finds good solutions | Loss < 0.001 for staccato, legato, grace note targets (Exp. 3) |
| Single-voice cannot match conflicting-regime targets | 3/4 parameters saturate at bounds; loss = 0.0196 (Exp. 4) |
| Two-voice achieves 21× lower loss | Loss = 0.00093; gap region improves 2888× (Exp. 4) |
| Transcription bottleneck is fragmentation, not recall | Baseline recall = 97%, precision = 21% (Application) |
| Stabilization yields order-of-magnitude F1 improvement | F1: 0.054 → 0.731 on piano (Application) |

The core message is simple: **excitable dynamics with globally-coupled parameters have strict expressivity limits, and those limits matter in practice.**