"""Visualization utilities for MIL-Lab models"""

from .heatmap_utils import (
    MILLabVisualizer,
    WholeSlideImage,
    draw_heatmap,
    sample_top_patches,
    to_percentiles,
)

from .trident_compat import (
    TridentVisualizer,
    visualize_heatmap_trident,
)

__all__ = [
    'MILLabVisualizer',
    'WholeSlideImage',
    'draw_heatmap',
    'sample_top_patches',
    'to_percentiles',
    'TridentVisualizer',
    'visualize_heatmap_trident',
]
