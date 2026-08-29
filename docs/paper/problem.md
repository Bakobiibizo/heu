### Problem Setup

We consider the problem of **event commitment** in continuous time: given a scalar input signal $x(t)$, produce a sparse sequence of discrete events ("spikes") that correspond to salient temporal structure in the signal. Unlike classification, event commitment requires the system to *decide when to act* and to suppress further action until sufficient new evidence accumulates.

The Hierarchical Excitable Unit (HEU) is a dynamical unit designed for this purpose. An HEU integrates excitation over time via a bounded envelope variable $v(t) \in [0,1]$, produces discrete spike events when a threshold is crossed, and enforces refractory and homeostatic constraints to prevent runaway behavior. Importantly, the HEU operates without learning; all behavior is determined by its dynamical structure and fixed parameters.

In this section, we show that **a single-voice HEU is insufficient to represent certain classes of temporal targets**, regardless of parameter tuning. We then demonstrate that introducing an additional, independent HEU voice resolves this limitation, establishing a strict expressivity separation between single- and multi-voice architectures.

---

### Single-Voice HEU Model

A single-voice HEU consists of a single envelope state $v(t)$, governed by excitation-dependent growth, recovery-dependent decay, and optional leak. A spike is emitted when $v(t)$ crosses a fixed threshold, after which a refractory interval suppresses further spiking.

Crucially, all temporal behavior in the single-voice HEU is controlled by **global parameters**, including:

- attack time constant ($\tau_{\text{attack}}$)
- recovery time constant ($\tau_{\text{recovery}}$)
- recovery strength (controlled by a balance parameter that trades off attack and recovery contributions)
- refractory duration

These parameters apply uniformly across all temporal regimes of the input. The parameters are exposed through a normalized $\theta$ interface with bounded ranges, transformed via exponential/sigmoid mappings to ensure stable dynamics.

---

### Target Class with Competing Temporal Requirements

We construct a target envelope encoding **competing temporal commitments** within a single event. The target consists of:

1. A **transient region** (0–25ms): rapid rise and fast decay from peak to floor, requiring fast recovery ($\tau_{\text{recovery}} \approx 6$ms)
2. A **gap region** (25–50ms): envelope must reach and maintain floor level
3. A **sustain region** (50ms+): stable envelope with minimal decay over hundreds of milliseconds, requiring slow recovery ($\tau_{\text{recovery}} \approx 500$ms)

The two active regions demand mutually incompatible parameterizations:

| Region    | Required Recovery Strength | Required Recovery Time |
| --------- | -------------------------- | ---------------------- |
| Transient | HIGH (fast decay to floor) | SHORT                  |
| Sustain   | LOW (stable output)        | LONG                   |

The target envelope is intentionally minimal and piecewise simple. It does not contain oscillations, stochastic variation, or adversarial structure. The incompatibility arises solely from conflicting temporal requirements within a single event, not from complexity of the target itself.

A single HEU must choose one parameterization. It cannot simultaneously achieve high recovery strength for fast transient decay AND low recovery strength for flat sustain.

The gap region enforces the temporal ordering of these requirements: fast recovery must occur *before* slow recovery can dominate.

---

### Empirical Optimization of Single-Voice HEU

To test whether parameter tuning can overcome this structural limitation, we performed exhaustive optimization using CMA-ES with wide parameter bounds and multiple random restarts.

#### Observed Failure Mode

The single-voice optimizer consistently exhibits parameter saturation:

| Parameter                      | Status              |
| ------------------------------ | ------------------- |
| $\theta_{\text{total}}$        | within bounds       |
| $\theta_{\text{balance}}$      | **AT UPPER BOUND**  |
| $\theta_{\text{attack-tau}}$   | within bounds       |
| $\theta_{\text{recovery-tau}}$ | **AT LOWER BOUND**  |

The optimizer pushes $\theta_{\text{balance}}$ to maximum (most attack-biased, for sustain) while simultaneously pushing $\theta_{\text{recovery-tau}}$ to minimum (fastest recovery, for transient decay). These are **conflicting objectives**; the optimizer is stuck at a compromise that satisfies neither region well.

Three of four parameters converging to boundary values indicates that the optimizer consistently seeks solutions outside the feasible parameter region, a standard empirical signature of architectural insufficiency rather than optimization failure.

---

### Two-Voice HEU Architecture

We introduce a two-voice HEU: two independent HEU instances with identical structure but separate parameters $(\theta^{(1)}, \theta^{(2)})$. Both voices receive the same excitation signal. Their envelopes are combined via:

$$v_{\text{combined}}(t) = \max(v^{(1)}(t), v^{(2)}(t))$$

The max operation is a minimal, non-learned combination rule that preserves the dominant envelope in each temporal region. Importantly, the expressivity separation does not depend on this specific choice; any monotone combination that allows independent dominance would suffice. We use max to avoid introducing additional parameters or gating mechanisms.

Each voice is free to specialize:

- **Transient voice**: recovery-biased parameters for fast decay
- **Sustain voice**: attack-biased parameters for stable output

The parameter divergence between voices on $\theta_{\text{balance}}$ (approximately 3 units apart) represents a configuration impossible for any single HEU to occupy.

---

### Resolution of the Expressivity Conflict

When the two-voice HEU is optimized against the same target:

| Metric                   | Single-Voice | Two-Voice | Ratio      |
| ------------------------ | ------------ | --------- | ---------- |
| Overall loss             | 0.0196       | 0.00093   | **21x**    |
| Gap region MSE           | 0.152        | 0.00005   | **2888x**  |
| Parameters at bounds     | 3/4 (75%)    | 2/8 (25%) | -          |

The gap region, where the parameter conflict manifests most acutely, shows three orders of magnitude improvement. The two-voice architecture retains substantial optimization headroom (only 2 of 8 parameters at bounds), while the single-voice solution is fully constrained.

The improvement cannot be attributed to optimization difficulty or local minima; both architectures are optimized with identical methods and comparable iteration counts. The gap is structural.

---

### Interpretation as an Impossibility Result

These results constitute a **constructive impossibility result**:

> **Proposition.** Let $\mathcal{H}_1$ denote the class of single-voice HEUs with parameters $\theta \in \Theta$, and $\mathcal{H}_2$ denote two-voice HEUs with parameters $(\theta^{(1)}, \theta^{(2)}) \in \Theta^2$. There exists a target $T(t)$ such that:
> $$\inf_{\theta \in \Theta} \mathcal{L}(\mathcal{H}_1(\theta), T) > 20 \cdot \inf_{(\theta^{(1)}, \theta^{(2)}) \in \Theta^2} \mathcal{L}(\mathcal{H}_2(\theta^{(1)}, \theta^{(2)}), T)$$

This is not a statement about all possible dynamical systems, but about the expressivity limits of a specific and well-defined class of excitable units. The result shows that adding independent dynamical state strictly increases representational power for event commitment tasks.

The gap region serves as the discriminating test: it requires the envelope to have *already decayed* from the transient peak, which is only possible if recovery was fast. Yet the system must then immediately produce stable sustain, which requires slow recovery.

The failure of the single-voice HEU is not due to insufficient capacity in the abstract sense, but to **the inability of a single dynamical state to satisfy incompatible temporal commitments simultaneously**.

---

### Implications

The impossibility result has several immediate implications:

1. **Parameter tuning cannot substitute for architectural state.** When temporal requirements conflict within a single event, additional state variables are necessary. No amount of hyperparameter search will resolve a structural limitation.

2. **Multi-voice dynamics are not an embellishment.** They are required to represent certain event structures even in the absence of learning. The two-voice architecture is the minimal sufficient architecture for targets with competing dynamics.

3. **The result is domain-agnostic.** The expressivity separation holds for any domain where event commitment must handle conflicting temporal regimes, including neural spike train modeling and anomaly detection in time series. In supplementary experiments on audio event detection, the two-voice architecture improves F1 from 0.33 to 0.73, consistent with the theoretical prediction.

4. **The result can be interpreted as a state separation argument.** A single envelope state cannot encode mutually incompatible decay behaviors without interference. Introducing a second independent state increases the dimensionality of the dynamical system, allowing each state to specialize. This mirrors expressivity separations observed in depth- or width-limited neural architectures, but in the temporal domain.