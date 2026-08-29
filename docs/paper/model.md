## Model

### Overview

We study the **Hierarchical Excitable Unit (HEU)**, a continuous-time dynamical unit designed for sparse event commitment in scalar input signals. The HEU maps a continuous excitation signal $x(t)$ to a bounded internal envelope $v(t) \in [0,1]$ and produces discrete spike events when sufficient evidence accumulates.

The term "hierarchical" refers to the intended use of HEUs within layered architectures, where lower layers handle fast dynamics and higher layers modulate parameters or aggregate outputs. This paper focuses on the single-unit case; hierarchical composition is addressed in future work.

The HEU is **non-learning**: all behavior is determined by fixed parameters and state evolution. Its design explicitly separates:

1. evidence accumulation,
2. state-dependent decay,
3. spike emission,
4. post-spike refractoriness,
5. long-term activity regulation.

We use "evidence" descriptively to denote accumulated excitation, without probabilistic interpretation.

This section formally defines the single-voice HEU model and its parameterization.

---

### Continuous-Time Dynamics

Let $x(t) \in \mathbb{R}_{\ge 0}$ denote the excitation signal. The excitation signal may contain both transient (pulsed) components and tonic (sustained) components; the HEU dynamics treat both uniformly through the same accumulation mechanism.

The HEU maintains a scalar envelope state $v(t) \in [0,1]$, whose dynamics are governed by competing growth and decay processes:

$$\frac{dv(t)}{dt} = \underbrace{G\big(x(t), v(t)\big)}_{\text{excitation-driven growth}} - \underbrace{R\big(v(t)\big)}_{\text{recovery-driven decay}} - \underbrace{L\big(v(t)\big)}_{\text{leak (optional)}}$$

where each term is bounded and state-dependent. All results hold for any bounded, monotone excitation transform in $G$ and any monotone recovery function in $R$. In experiments, we use sigmoid-shaped gate functions; full definitions are provided in Appendix A.

---

### Excitation-Driven Growth

The growth term integrates incoming excitation while enforcing saturation:

$$G(x, v) = \alpha(v) \cdot g(x) \cdot (1 - v)$$

where:

- $g(x)$ is a non-negative, bounded transform of the input excitation,
- $\alpha(v) > 0$ is the effective attack rate, modulated by envelope state,
- the factor $(1 - v)$ enforces saturation at $v = 1$.

This form ensures that growth diminishes as the envelope approaches its upper bound.

---

### Recovery-Driven Decay

The decay term drives the envelope back toward zero:

$$R(v) = \beta(v) \cdot v$$

where $\beta(v) > 0$ is the recovery strength, modulated by envelope state.

Recovery is state-dependent: higher envelope values induce stronger decay, enabling rapid suppression after large excursions.

---

### Coupled Gate Dynamics and Balance Constraint

In the HEU, excitation-driven growth and recovery-driven decay are not independently parameterized. Both processes are gated by state-dependent activations that share a common balance parameter $\theta_{\text{balance}}$. This parameter controls the relative contribution of attack and recovery to the envelope derivative: attack-biased settings ($\theta_{\text{balance}} > 0$) favor sustained envelope values, while recovery-biased settings ($\theta_{\text{balance}} < 0$) favor rapid return to baseline.

Crucially, there is no parameterization that achieves both simultaneously; the tradeoff is structural. Increasing attack persistence necessarily weakens recovery, and increasing recovery strength necessarily reduces attack persistence. This coupling is the source of the expressivity limits demonstrated in the following section.

---

### Leak Term (Optional)

An optional leak term provides slow baseline decay:

$$L(v) = \lambda \cdot v$$

with $\lambda \ge 0$. In most experiments, $\lambda$ is small relative to recovery and does not qualitatively affect the results.

---

### Spike Emission

A spike event is emitted when the envelope crosses a fixed threshold:

$$\text{spike at time } t \quad \text{if} \quad v(t) \ge v_{\text{thresh}}$$

Spike emission does **not** reset the envelope state. Instead, it triggers a refractory mechanism that temporarily suppresses further spike emission.

This design decouples envelope dynamics from spike timing, allowing the envelope to encode evidence independently of event commitment.

---

### Refractory Mechanism

After a spike, the HEU enters a refractory period of duration $\tau_{\text{ref}}$, during which additional spikes are suppressed:

$$\text{spike allowed} \iff \text{refractory\_remaining}(t) = 0$$

The refractory state evolves deterministically as a countdown timer. Importantly, refractoriness **does not alter** the envelope dynamics $v(t)$; it only gates spike emission.

This separation allows the HEU to accumulate evidence during refractory periods without committing additional events.

---

### Homeostatic Regulation

To ensure long-term stability across varying excitation statistics, the HEU tracks a slow exponential moving average of activity:

$$a(t) = \text{EMA}\big(\text{spikes}(t)\big)$$

When activity exceeds a target range, a damping factor reduces effective excitation gain. Homeostasis acts on a timescale orders of magnitude slower than envelope dynamics and does not affect short-term behavior.

The impossibility result in the following section holds even when homeostatic regulation is disabled; homeostasis is included for completeness and long-term stability in downstream applications.

---

### Parameterization via Normalized Coordinates

All HEU parameters are expressed through a normalized vector

$$\theta \in \Theta \subset \mathbb{R}^d$$

with bounded components.

Each component of $\theta$ is mapped to a physical parameter (e.g., $\tau_{\text{attack}}$, $\tau_{\text{recovery}}$, $\theta_{\text{balance}}$) via exponential or sigmoid transforms. This ensures:

- positivity where required,
- bounded dynamics,
- numerical stability during optimization.

Crucially, **all parameters are global**: a single parameterization governs behavior across all temporal regimes of the input.

---

### Single-Voice Assumption

A *single-voice HEU* maintains exactly one envelope state $v(t)$ and one parameter vector $\theta$. All excitation, regardless of temporal context, must be mediated through this single state.

In the following section, we demonstrate that this single-state, globally-parameterized architecture imposes strict expressivity limits when temporal requirements conflict.

---

### Multi-Voice Extension (Preview)

A multi-voice HEU consists of multiple independent HEU instances receiving the same excitation signal, each with its own state and parameters. Envelope combination and spike emission are defined externally.

The impossibility result in the following section demonstrates that even a minimal two-voice extension strictly increases representational capacity.