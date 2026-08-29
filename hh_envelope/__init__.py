"""
HH-Envelope: HH-inspired envelope dynamics for musical motion.

A continuous-time dynamical envelope generator with:
- Fast self-amplifying transient behavior
- Delayed recovery and refractory spacing
- Interpretable parameters (attack strength, recovery strength, recovery time)

Modules:
    core: Core envelope unit and dynamics
    hierarchy: Multi-timescale layer management
    polyphony: Voice objects and coupling
    learning: Loss functions and optimization
    experiments: Hypothesis-driven validation experiments
    utils: Signal generation, metrics, and plotting
"""

__version__ = "0.1.0"
