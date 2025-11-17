"""
Trident-compatible visualization functions

This module provides visualization functions that work with Trident-extracted
patch coordinates, matching Trident's visualization behavior exactly.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
import os

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False


def create_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    region_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create heatmap overlay (Trident-compatible).

    Args:
        scores: Normalized scores (N,)
        coords: Patch coordinates at level 0 (N, 2)
        patch_size_level0: Patch size at level 0
        scale: Scaling factors [scale_x, scale_y]
        region_size: Region dimensions (w, h)

    Returns:
        Heatmap overlay array
    """
    patch_size = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
    coords_scaled = np.ceil(coords * scale).astype(int)

    overlay = np.zeros(tuple(np.flip(region_size)), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)

    for idx, coord in enumerate(coords_scaled):
        y_start = coord[1]
        y_end = coord[1] + patch_size[1]
        x_start = coord[0]
        x_end = coord[0] + patch_size[0]

        overlay[y_start:y_end, x_start:x_end] += scores[idx]
        counter[y_start:y_end, x_start:x_end] += 1

    # Average overlapping regions
    zero_mask = counter == 0
    overlay[~zero_mask] /= counter[~zero_mask]
    overlay[zero_mask] = np.nan  # No data regions

    return overlay


def apply_colormap(overlay: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Apply colormap to overlay.

    Args:
        overlay: Heatmap overlay with possible NaN values
        cmap_name: Matplotlib colormap name

    Returns:
        RGB colored overlay
    """
    cmap = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay)

    if valid_mask.any():
        colored_valid = (cmap(overlay[valid_mask]) * 255).astype(np.uint8)[:, :3]
        overlay_colored[valid_mask] = colored_valid

    return overlay_colored


def draw_numbered_markers(
    img: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    top_k_indices: np.ndarray,
    marker_size: int = 60,
    text_size: float = 2.0,
    thickness: int = 4
) -> np.ndarray:
    """
    Draw numbered markers on image at top-k patch locations.

    Args:
        img: Image array to draw on (will be modified)
        coords: Patch coordinates at level 0 (N, 2)
        patch_size_level0: Patch size at level 0
        scale: Scaling factors [scale_x, scale_y]
        top_k_indices: Indices of top-k patches to mark
        marker_size: Size of the marker circle
        text_size: Font scale for numbers
        thickness: Line thickness for circle and text

    Returns:
        Image with markers drawn
    """
    # Scale patch size and coordinates to visualization level
    patch_size = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
    coords_scaled = np.ceil(coords * scale).astype(int)

    for rank, idx in enumerate(top_k_indices):
        # Calculate center of patch
        x, y = coords_scaled[idx]
        center_x = int(x + patch_size[0] // 2)
        center_y = int(y + patch_size[1] // 2)

        # Draw black filled circle with white outline
        cv2.circle(img, (center_x, center_y), marker_size, (255, 255, 255), thickness + 4)  # White outline
        cv2.circle(img, (center_x, center_y), marker_size, (0, 0, 0), -1)  # Black filled circle

        # Draw number
        number = str(rank + 1)
        text_thickness = max(2, thickness - 1)

        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(
            number, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness
        )

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Draw black outline for text (for contrast)
        cv2.putText(img, number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness + 3, cv2.LINE_AA)

        # Draw white text (visible on black circle)
        cv2.putText(img, number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return img


def draw_patch_number(patch_img: Image.Image, number: int, position: str = 'top-left') -> Image.Image:
    """
    Draw a number on a patch image.

    Args:
        patch_img: PIL Image of the patch
        number: Number to draw (1, 2, 3, etc.)
        position: Position of the number ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')

    Returns:
        PIL Image with number drawn
    """
    # Convert to numpy for drawing
    img_array = np.array(patch_img)
    height, width = img_array.shape[:2]

    # Calculate marker size based on image size
    marker_size = int(min(width, height) * 0.12)
    text_size = min(width, height) / 150.0  # Smaller text (was 80.0)
    thickness = max(2, int(min(width, height) / 60))

    # Determine position
    if position == 'top-left':
        center_x, center_y = marker_size + 10, marker_size + 10
    elif position == 'top-right':
        center_x, center_y = width - marker_size - 10, marker_size + 10
    elif position == 'bottom-left':
        center_x, center_y = marker_size + 10, height - marker_size - 10
    elif position == 'bottom-right':
        center_x, center_y = width - marker_size - 10, height - marker_size - 10
    else:  # center
        center_x, center_y = width // 2, height // 2

    # Draw black filled circle with white outline
    cv2.circle(img_array, (center_x, center_y), marker_size, (255, 255, 255), thickness + 4)  # White outline
    cv2.circle(img_array, (center_x, center_y), marker_size, (0, 0, 0), -1)  # Black filled circle

    # Draw number (black text with white outline for visibility on black circle)
    number_str = str(number)
    text_thickness = max(2, thickness - 1)

    # Get text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(
        number_str, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness
    )

    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2

    # Draw black outline for text (for contrast)
    cv2.putText(img_array, number_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                text_size, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)

    # Draw white text (visible on black circle)
    cv2.putText(img_array, number_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                text_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return Image.fromarray(img_array)


def visualize_heatmap_trident(
    wsi_path: str,
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int = 256,
    vis_level: int = -1,
    cmap: str = 'jet',
    normalize: bool = True,
    alpha: float = 0.4,
    output_path: Optional[str] = None,
    num_top_patches: int = 0,
    top_patches_dir: Optional[str] = None,
    mark_top_patches: bool = False,
    marker_size_ratio: float = 0.015,
    text_size_ratio: float = 0.008,
    marker_size: int = None,  # Deprecated: use marker_size_ratio
    text_size: float = None   # Deprecated: use text_size_ratio
) -> Image.Image:
    """
    Generate heatmap visualization compatible with Trident coordinates.

    This function matches Trident's visualization approach:
    - No tissue segmentation at visualization time
    - Rank-based normalization
    - Simple alpha blending

    Args:
        wsi_path: Path to WSI file
        scores: Attention scores (N,)
        coords: Patch coordinates at level 0 (N, 2)
        patch_size_level0: Patch size at level 0
        vis_level: Visualization pyramid level (-1 = auto)
        cmap: Matplotlib colormap name
        normalize: Use rank-based normalization (like Trident)
        alpha: Heatmap opacity (0=transparent, 1=opaque)
        output_path: Path to save heatmap
        num_top_patches: Number of top patches to save
        top_patches_dir: Directory to save top patches
        mark_top_patches: If True, draw numbered markers on heatmap for top patches
        marker_size_ratio: Marker radius as ratio of image size (default: 0.015)
                          Examples: 0.01=small, 0.015=medium, 0.02=large
        text_size_ratio: Text size as ratio of image size (default: 0.008)
                        Examples: 0.005=small, 0.008=medium, 0.012=large
        marker_size: (Deprecated) Fixed pixel size for markers. Use marker_size_ratio instead.
        text_size: (Deprecated) Fixed font scale. Use text_size_ratio instead.

    Returns:
        PIL Image with heatmap overlay
    """
    if not OPENSLIDE_AVAILABLE:
        raise ImportError("openslide-python required. Install with: pip install openslide-python")

    # Open WSI
    wsi = openslide.open_slide(wsi_path)

    # Auto-select visualization level
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    downsample = wsi.level_downsamples[vis_level]
    scale = np.array([1 / downsample, 1 / downsample])
    level0_dims = wsi.level_dimensions[0]
    region_size = tuple((np.array(level0_dims) * scale).astype(int))

    print(f"\nCreating Trident-style heatmap:")
    print(f"  WSI level 0 dimensions: {level0_dims}")
    print(f"  Visualization level: {vis_level}")
    print(f"  Downsample: {downsample}")
    print(f"  Region size: {region_size}")
    print(f"  Number of patches: {len(coords)}")

    # Normalize scores (Trident uses rank-based normalization)
    if normalize:
        from scipy.stats import rankdata
        scores_norm = rankdata(scores, 'average') / len(scores)
        print(f"  Score range (after rank normalization): [{scores_norm.min():.4f}, {scores_norm.max():.4f}]")
    else:
        scores_norm = scores
        print(f"  Score range (raw): [{scores_norm.min():.4f}, {scores_norm.max():.4f}]")

    # Create overlay
    print("  Creating overlay...")
    overlay = create_overlay(scores_norm, coords, patch_size_level0, scale, region_size)

    # Normalize overlay to [0, 1] for colormap
    valid_mask = ~np.isnan(overlay)
    if valid_mask.any():
        overlay_min = overlay[valid_mask].min()
        overlay_max = overlay[valid_mask].max()
        if overlay_max > overlay_min:
            overlay[valid_mask] = (overlay[valid_mask] - overlay_min) / (overlay_max - overlay_min)
        else:
            overlay[valid_mask] = 0.5

    # Apply colormap
    print("  Applying colormap...")
    overlay_colored = apply_colormap(overlay, cmap)

    # Read background image
    print("  Reading background image...")
    img = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)

    # Blend overlay with background
    print(f"  Blending (alpha={alpha})...")
    blended = cv2.addWeighted(img, 1 - alpha, overlay_colored, alpha, 0)

    # Draw numbered markers for top patches if requested
    if mark_top_patches and num_top_patches > 0:
        print(f"  Drawing numbered markers for top-{num_top_patches} patches...")
        topk_indices = np.argsort(scores)[-num_top_patches:][::-1]

        # Calculate marker and text sizes based on image dimensions
        img_size = min(region_size[0], region_size[1])

        # Handle backward compatibility: if old parameters provided, use them
        if marker_size is not None:
            print(f"  ⚠️  Using deprecated parameter 'marker_size'. Consider using 'marker_size_ratio' instead.")
            final_marker_size = marker_size
        else:
            final_marker_size = int(img_size * marker_size_ratio)

        if text_size is not None:
            print(f"  ⚠️  Using deprecated parameter 'text_size'. Consider using 'text_size_ratio' instead.")
            final_text_size = text_size
        else:
            final_text_size = img_size * text_size_ratio

        print(f"    Image size: {region_size}, Marker size: {final_marker_size}px, Text size: {final_text_size:.2f}")

        blended = draw_numbered_markers(
            blended, coords, patch_size_level0, scale, topk_indices,
            marker_size=final_marker_size, text_size=final_text_size
        )

    blended_img = Image.fromarray(blended)

    # Save heatmap
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        blended_img.save(output_path)
        print(f"  Saved to: {output_path}")

    # Save top-k patches
    if num_top_patches > 0 and top_patches_dir:
        print(f"  Saving top-{num_top_patches} patches...")
        os.makedirs(top_patches_dir, exist_ok=True)
        topk_indices = np.argsort(scores)[-num_top_patches:][::-1]

        for rank, idx in enumerate(topk_indices):
            x, y = coords[idx]
            patch = wsi.read_region((int(x), int(y)), 0, (patch_size_level0, patch_size_level0))
            patch = patch.convert('RGB')

            # Draw number on patch if markers are enabled
            if mark_top_patches:
                patch = draw_patch_number(patch, rank + 1, position='top-left')

            patch_path = os.path.join(top_patches_dir, f"top_{rank+1}_score_{scores[idx]:.4f}_x{int(x)}_y{int(y)}.png")
            patch.save(patch_path)

        print(f"  Saved top patches to: {top_patches_dir}")

    wsi.close()
    print("  Done!")

    return blended_img


class TridentVisualizer:
    """
    High-level Trident-compatible visualizer for MIL-Lab models.

    Example:
        >>> viz = TridentVisualizer(model, wsi_path='slide.tiff')
        >>> heatmap = viz.create_heatmap(
        ...     features=features,
        ...     coords=coords,
        ...     cmap='jet',
        ...     alpha=0.4
        ... )
        >>> heatmap.save('heatmap.png')
    """

    def __init__(self, model, wsi_path: str):
        """
        Initialize Trident-compatible visualizer.

        Args:
            model: MIL-Lab model
            wsi_path: Path to WSI file
        """
        self.model = model
        self.wsi_path = wsi_path

    def create_heatmap(
        self,
        features,
        coords,
        attention_scores=None,
        patch_size_level0: int = 256,
        vis_level: int = -1,
        cmap: str = 'jet',
        alpha: float = 0.4,
        normalize: bool = True,
        **kwargs
    ) -> Image.Image:
        """
        Create heatmap using Trident-compatible visualization.

        Args:
            features: Patch features (N, D) or (1, N, D)
            coords: Patch coordinates (N, 2)
            attention_scores: Pre-computed scores (if None, computed from model)
            patch_size_level0: Patch size at level 0
            vis_level: Visualization level
            cmap: Colormap name
            alpha: Heatmap opacity
            normalize: Use rank-based normalization
            **kwargs: Additional arguments for visualize_heatmap_trident

        Returns:
            PIL Image with heatmap
        """
        import torch

        # Compute attention if not provided
        if attention_scores is None:
            self.model.eval()
            with torch.no_grad():
                if len(features.shape) == 2:
                    features = features.unsqueeze(0)

                results_dict, log_dict = self.model(features.cuda(), return_attention=True)
                attention_scores = log_dict['attention'].cpu().numpy().squeeze()

        # Ensure matching lengths
        min_len = min(len(coords), len(attention_scores))
        coords = coords[:min_len]
        attention_scores = attention_scores[:min_len]

        # Create heatmap
        return visualize_heatmap_trident(
            wsi_path=self.wsi_path,
            scores=attention_scores,
            coords=coords,
            patch_size_level0=patch_size_level0,
            vis_level=vis_level,
            cmap=cmap,
            alpha=alpha,
            normalize=normalize,
            **kwargs
        )
