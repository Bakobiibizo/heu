"""
Hierarchy module: Multi-scale layer management.

Priority: P3

Three layers with different time scales:
- Layer 1: Note-level (fast) - handles transient articulation
- Layer 2: Phrase-level (medium) - parameter modulation to note-level
- Layer 3: Form-level (slow) - long-term tension/density control

Critical rule: Higher layers modulate PARAMETERS, not direct signal injection.
"""

from .layer_stack import LayerStack, LayerStackOutput

__all__ = ["LayerStack", "LayerStackOutput"]
