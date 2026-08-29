## Related Work

The HEU draws on four research threads: biophysical neuron models, spiking neural networks, automatic music transcription, and expressivity analysis in neural architectures. We position our contribution relative to each.

---

### Hodgkin-Huxley and Excitable Systems

The Hodgkin-Huxley (HH) model describes action potential generation through voltage-gated ion channel dynamics. The original formulation uses four coupled differential equations governing membrane potential and three gating variables. Subsequent simplifications—FitzHugh-Nagumo, Morris-Lecar, Izhikevich—reduce dimensionality while preserving qualitative excitable behavior: threshold-triggered spikes, refractory periods, and adaptation.

The HEU belongs to this lineage but differs in purpose. Where HH-family models aim for biological fidelity (reproducing spike shapes, frequency-current curves, bursting patterns), the HEU aims for **signal processing utility**. We retain the core excitable structure—competing growth and decay processes, threshold commitment, refractory suppression—but parameterize for interpretability and optimization rather than biophysical correspondence.

Specifically, the HEU uses:
- Sigmoid gate dynamics rather than Boltzmann activation functions
- A normalized parameter interface (θ ∈ [−2, 2]⁴) rather than conductances in mS/cm²
- Envelope output rather than membrane voltage
- No channel-specific kinetics (Na⁺, K⁺, leak are abstracted into attack/recovery balance)

This makes the HEU amenable to gradient-free optimization (CMA-ES) and compositional architectures, at the cost of biological interpretability.

---

### Spiking Neural Networks

Spiking neural networks (SNNs) use discrete spike events as the primary information carrier. Common neuron models include leaky integrate-and-fire (LIF), adaptive exponential (AdEx), and spike response models (SRM). SNNs have been applied to temporal pattern recognition, neuromorphic computing, and energy-efficient inference.

The HEU is **not** a spiking neural network in the standard sense:
- It has no synaptic weights or learning rules
- It does not propagate spikes through a network topology
- Its "spikes" are output events, not inter-unit communication

The relationship is closer to the **single-neuron modeling** literature than to SNN architectures. Our impossibility result concerns the expressivity of individual excitable units, not network-level computation. The finding that single-voice dynamics have strict limits complements SNN research showing that recurrent network topology and spike timing codes expand representational capacity.

Recent work on surrogate gradient training for SNNs addresses the non-differentiability of spike generation. The HEU sidesteps this by using gradient-free optimization and treating spikes as discrete outputs rather than differentiable signals. Whether surrogate gradient methods could improve HEU parameter fitting is an open question we do not address.

---

### Automatic Music Transcription

Automatic music transcription (AMT) converts audio to symbolic representations (typically MIDI). The task decomposes into pitch detection, onset detection, and note tracking. Modern systems use deep learning: convolutional networks for spectrogram processing, recurrent networks for temporal modeling, and attention mechanisms for long-range dependencies.

State-of-the-art AMT systems achieve F1 > 0.85 on standard benchmarks (MAPS, MAESTRO) using models with millions of parameters trained on thousands of hours of audio. The HEU-based transcriber we present is **not competitive** with these systems and is not intended to be.

Our contribution to AMT is diagnostic, not methodological:
1. We identify **note fragmentation** (over-detection at correct pitch) as the dominant failure mode when excitable dynamics operate at coarse temporal resolution.
2. We show that this fragmentation is a manifestation of the single-voice expressivity limit—the same parameter conflict that causes 21× worse loss in controlled experiments causes 3–5× over-detection in transcription.
3. We demonstrate that stabilization (upsampling or hysteresis) resolves fragmentation, improving F1 from 0.054 to 0.731 on piano content.

The finding that **recall is not the bottleneck** (baseline achieves 97% recall) contradicts common assumptions in AMT pipeline design, where onset detection is often treated as the primary challenge. For HEU-based systems, precision through fragmentation control is the limiting factor.

Prior work on onset detection includes signal processing methods (spectral flux, complex domain deviation) and learned approaches (CNNs on mel spectrograms). The HEU operates downstream of onset detection—it converts continuous pitch likelihood into discrete note events. The fragmentation problem arises not from missing onsets but from the dynamics creating spurious additional onsets.

---

### Expressivity and Representational Capacity

Expressivity analysis characterizes what functions a model class can and cannot represent. Classical results include universal approximation theorems for neural networks and depth separation results showing that deep networks can represent functions requiring exponentially wide shallow networks.

Our impossibility result is a **separation result** in this tradition: we exhibit a target that single-voice HEUs cannot represent while two-voice HEUs can. The separation is:
- **Constructive**: we provide the target explicitly
- **Quantitative**: 21× loss ratio, 2888× in the critical region
- **Mechanistic**: we identify parameter coupling as the cause

The result differs from typical neural network expressivity analysis in several ways:
1. The model class is much smaller (4 parameters vs. millions)
2. The separation is between 1 and 2 instances, not asymptotic in width/depth
3. The failure mode is parameter saturation, not approximation error

Related separation results in dynamical systems include work showing that certain temporal patterns require memory (hidden state) that systems with insufficient independent dynamical state cannot provide. Our result is similar in spirit: the single-voice HEU lacks sufficient state to represent conflicting temporal regimes, and adding a second voice provides that state.

The "balance parameter" coupling in the HEU—where attack and recovery strength trade off—is analogous to capacity bottlenecks in other architectures. Mixture-of-experts models address similar issues by routing inputs to specialized sub-networks; the two-voice HEU can be viewed as a minimal mixture where each voice specializes in a temporal regime.

---

### Summary

| Thread | HEU Relationship |
|--------|------------------|
| Hodgkin-Huxley models | Inspired by, but optimized for signal processing not biological fidelity |
| Spiking neural networks | Single-unit expressivity, not network computation; no learning |
| Automatic music transcription | Diagnostic application, not SOTA method; identifies fragmentation bottleneck |
| Expressivity analysis | Constructive separation result; parameter coupling limits single-voice capacity |

The HEU occupies a specific niche: a minimal, interpretable excitable system whose expressivity limits can be characterized precisely. It is not a replacement for learned models in applications requiring high accuracy, but it provides a tractable setting for understanding when and why additional dynamical state is necessary.