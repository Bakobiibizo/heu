## Discussion

We have shown that a single-voice HEU cannot represent temporal targets with conflicting dynamical requirements, while a minimal two-voice architecture succeeds. This section interprets the result, addresses limitations, and suggests extensions.

---

### Interpretation

The impossibility result is fundamentally about **parameter coupling**. The HEU's attack and recovery processes share a balance parameter that determines their relative strength. This coupling is not incidental—it reflects a design choice to keep the parameter space small and interpretable. But it creates a tradeoff: settings that favor fast transient decay necessarily weaken sustained output, and vice versa.

When a target requires both behaviors in sequence (transient → gap → sustain), the optimizer faces an impossible choice. It cannot tune for fast decay without sacrificing sustain, or tune for stable sustain without sacrificing decay. The result is parameter saturation: the optimizer pushes parameters to their bounds, seeking configurations outside the feasible region.

The two-voice architecture resolves this by **decoupling** the conflicting requirements. Each voice has independent parameters and can specialize: one for transient dynamics, one for sustained dynamics. The max combination rule selects the dominant voice in each temporal region. No learning is required—the separation emerges from optimization against the target.

This pattern—parameter coupling creating expressivity limits that additional state resolves—is expected to arise in other systems with globally coupled temporal parameters. Any system where a single parameter governs multiple temporal behaviors will face similar conflicts when those behaviors must coexist in a single output.

---

### What the Result Does and Does Not Show

**The result shows:**
- Single-voice HEUs have strict expressivity limits for certain targets
- The limit manifests as parameter saturation (3/4 at bounds)
- Two-voice architecture achieves 21× lower loss on such targets
- The gap region (where conflict is acute) shows 2888× improvement
- The same bottleneck appears in practice (transcription fragmentation)

**The result does not show:**
- That all targets require multiple voices (most do not)
- That two voices are always sufficient (more complex targets may need more)
- That the HEU is optimal for any application
- That the 21× ratio is tight (it may be possible to construct harder targets)
- That learned systems face the same limits (they have far more parameters)

The impossibility is **constructive and specific**, not universal. We exhibit one target that separates single-voice from two-voice; we do not characterize the full class of such targets.

---

### Limitations

**Limited target class.** The impossibility target is hand-constructed to create parameter conflict. Real-world signals may or may not contain such conflicts. The transcription application suggests conflicts do arise in practice (fragmentation), but we have not systematically characterized which musical patterns trigger single-voice failure.

**Small evaluation scale.** The transcription experiments use 9 audio files. This is sufficient for proof-of-concept but insufficient for generalization claims. Performance on larger, more diverse datasets remains untested.

**No learned baselines.** We compare single-voice to two-voice HEUs, not to learned alternatives. A fair comparison would require training neural networks on the same targets, which we have not done. The HEU's advantage is interpretability and minimal parameterization, not raw performance.

**Coupling experiment inconclusive.** Experiment 2 validates that inter-voice coupling does not cause instability, but does not demonstrate that coupling produces meaningful differentiation. The experimental regime (non-overlapping envelopes) renders coupling inert. Whether coupling provides benefit under appropriate conditions remains open.

**CQT frontend limits bass detection.** The transcription system fails completely on bass content (F1 = 0) due to spectral leakage in the CQT at low frequencies. This is a frontend limitation unrelated to the HEU, but it restricts the practical applicability of the current system.

**Velocity encoding destroyed upstream.** Per-frame normalization in the harmonic sum eliminates absolute energy information before it reaches the HEU. The HEU envelope therefore cannot encode musical dynamics (velocity), despite this being a stated motivation. Preserving dynamics would require architectural changes to the pitch detection stage.

---

### Extensions

**Characterizing the separating target class.** What properties make a target impossible for single-voice? We conjecture that the critical case is targets requiring disjoint regions of θ-space within a single event—parameter settings that cannot be simultaneously satisfied. Formalizing this would strengthen the impossibility result.

**Scaling to more voices.** The two-voice architecture is minimal. For targets with three or more conflicting regimes, additional voices may be needed. Is there a systematic relationship between target complexity and required voice count?

**Learned combination rules.** We use max combination, which is parameter-free but rigid. A learned combination (e.g., attention-weighted mixture) might improve performance on complex targets. This would sacrifice some interpretability for flexibility.

**Coupling under overlap conditions.** The coupling mechanism modulates attack gain based on partner envelope state. Testing under conditions where envelopes overlap (fast tempo, sustained tonic) would determine whether coupling provides meaningful musical benefit (e.g., consonance-dependent response shaping).

**Alternative frontends for bass.** Multi-resolution CQT, reassigned spectrograms, or dedicated low-frequency pitch tracking could address the bass detection failure. This is orthogonal to the HEU contribution but necessary for a complete transcription system.

**Integration with learned systems.** The HEU could serve as a structured prior or inductive bias within a larger learned architecture. For example, an HEU bank could provide interpretable note-level dynamics while a neural network handles pitch detection. Whether this improves over end-to-end learning is an empirical question.

---

### Broader Implications

The impossibility result has implications beyond music and beyond the HEU specifically:

**For dynamical systems design:** Parameter coupling creates expressivity limits. When designing systems for temporal pattern representation, coupling decisions constrain what patterns are representable. Adding independent state (voices, channels, memory) expands capacity but increases complexity.

**For neural architecture analysis:** The single-voice vs. two-voice separation is analogous to depth and width separations in neural networks. Our result shows that even very small systems (4 parameters) can exhibit strict expressivity limits that additional structure resolves. This suggests that expressivity analysis at the single-unit level may yield insights complementary to network-level analysis.

**For event detection broadly:** The fragmentation problem—high recall but low precision due to spurious re-triggering—likely affects any threshold-based or refractory-limited event detector operating at coarse temporal resolution. The HEU makes this failure mode explicit and provides a framework for understanding when it arises (parameter conflict between onset sensitivity and sustain stability).

---

### Conclusion of Discussion

The HEU is a minimal system that makes a maximal point: **globally coupled parameters impose strict expressivity limits on temporal pattern representation**. The limit is not asymptotic or approximate; it manifests as measurable parameter saturation and order-of-magnitude performance gaps. Adding independent dynamical state—even just one additional voice—resolves the limit for the targets we study.

This does not mean the HEU is the right tool for practical applications. Learned systems with millions of parameters will outperform it on any benchmark. But the HEU provides a tractable setting for understanding *why* certain architectures fail and *what* structural changes resolve the failure. That understanding may transfer to systems where interpretability is harder to achieve.