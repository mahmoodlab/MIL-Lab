"""
Advanced Heatmap Visualization for MIL-Lab Models

This module provides CLAM-inspired visualization utilities for MIL models,
including tissue-masked heatmap overlays, percentile normalization, and
top-k patch sampling.

Inspired by: CLAM (https://github.com/mahmoodlab/CLAM)
Adapted for: MIL-Lab framework
"""

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import h5py
import os
from typing import Optional, Tuple, Union, List
from pathlib import Path

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: openslide not available. Install with: pip install openslide-python")


def to_percentiles(scores):
    """Convert scores to percentiles for better contrast"""
    scores = scores.copy()
    scores_sorted = np.sort(scores.flatten())
    percentiles = np.array([percentileofscore(scores_sorted, score) for score in scores.flatten()])
    return percentiles.reshape(scores.shape)


def screen_coords(scores, coords, top_left, bot_right):
    """Filter coordinates and scores within bounding box"""
    mask = (coords[:, 0] >= top_left[0]) & (coords[:, 0] <= bot_right[0]) & \
           (coords[:, 1] >= top_left[1]) & (coords[:, 1] <= bot_right[1])
    return scores[mask], coords[mask]


class WholeSlideImage:
    """Wrapper for whole slide image with tissue segmentation and visualization"""

    def __init__(self, wsi_path: str):
        """
        Initialize WholeSlideImage object

        Args:
            wsi_path: Path to whole slide image file
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("openslide-python required. Install with: pip install openslide-python")

        self.name = os.path.splitext(os.path.basename(wsi_path))[0]
        self.wsi = openslide.open_slide(wsi_path)
        self.level_downsamples = self._get_level_downsamples()
        self.level_dim = self.wsi.level_dimensions
        self.contours_tissue = None
        self.holes_tissue = None

    def _get_level_downsamples(self):
        """Get downsample factors for each pyramid level"""
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        for dim in self.wsi.level_dimensions:
            level_downsamples.append((float(dim_0[0]) / dim[0], float(dim_0[1]) / dim[1]))
        return level_downsamples

    def segment_tissue(self, seg_level: int = -1, sthresh: int = 20,
                      mthresh: int = 7, close: int = 0, use_otsu: bool = False,
                      filter_params: dict = {'a_t': 100, 'a_h': 16, 'max_n_holes': 10}):
        """
        Segment tissue regions using HSV thresholding

        Args:
            seg_level: Pyramid level for segmentation (-1 = auto)
            sthresh: Saturation threshold
            mthresh: Median blur kernel size
            close: Morphological closing kernel size
            use_otsu: Use Otsu's method for thresholding
            filter_params: Parameters for contour filtering
        """
        if seg_level < 0:
            seg_level = self.wsi.get_best_level_for_downsample(32)

        # Read image at segmentation level
        img = np.array(self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)

        # Thresholding
        if use_otsu:
            _, img_thresh = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_thresh = cv2.threshold(img_med, sthresh, 255, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        # Filter contours by area
        scale = self.level_downsamples[seg_level]
        foreground_contours, hole_contours = self._filter_contours(
            contours, hierarchy, filter_params, scale
        )

        # Scale contours back to level 0
        self.contours_tissue = self._scale_contours(foreground_contours, scale)
        self.holes_tissue = [self._scale_contours(holes, scale) for holes in hole_contours]

    def _filter_contours(self, contours, hierarchy, filter_params, scale):
        """Filter contours by area"""
        filtered = []
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        ref_patch_size = 512
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        a_t = filter_params['a_t'] * scaled_ref_patch_area
        a_h = filter_params['a_h'] * scaled_ref_patch_area

        for cont_idx in hierarchy_1:
            cont = contours[cont_idx]
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            a = cv2.contourArea(cont)
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            a = a - np.array(hole_areas).sum()

            if a >= a_t:
                filtered.append(cont_idx)

                # Filter holes
                unfiltered_holes = [contours[idx] for idx in holes]
                unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfiltered_holes = unfiltered_holes[:filter_params['max_n_holes']]
                filtered_holes = [hole for hole in unfiltered_holes if cv2.contourArea(hole) > a_h]
                all_holes.append(filtered_holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]
        return foreground_contours, all_holes

    def _scale_contours(self, contours, scale):
        """Scale contours from one level to another"""
        return [np.array(cont * np.array([1/scale[0], 1/scale[1]]), dtype=np.int32) for cont in contours]

    def get_seg_mask(self, region_size, scale, use_holes=True, offset=(0, 0)):
        """
        Generate tissue segmentation mask

        Args:
            region_size: Size of the region (w, h)
            scale: Scaling factors
            use_holes: Whether to mask out holes
            offset: Offset from top-left

        Returns:
            Boolean mask indicating tissue regions
        """
        tissue_mask = np.zeros(region_size[::-1], dtype=np.uint8)

        if self.contours_tissue is None:
            return tissue_mask > 0

        offset = tuple(np.array(offset) * np.array(scale))

        for cont in self.contours_tissue:
            cont_scaled = (cont * np.array(scale)).astype(np.int32)
            cont_scaled = cont_scaled - np.array(offset).astype(np.int32)
            cv2.drawContours(tissue_mask, [cont_scaled], -1, 1, -1)

        if use_holes and self.holes_tissue is not None:
            for holes in self.holes_tissue:
                for hole in holes:
                    hole_scaled = (hole * np.array(scale)).astype(np.int32)
                    hole_scaled = hole_scaled - np.array(offset).astype(np.int32)
                    cv2.drawContours(tissue_mask, [hole_scaled], -1, 0, -1)

        return tissue_mask > 0


def draw_heatmap(
    scores: np.ndarray,
    coords: np.ndarray,
    slide_path: str,
    wsi_object: Optional[WholeSlideImage] = None,
    vis_level: int = -1,
    patch_size: Tuple[int, int] = (256, 256),
    cmap: str = 'jet',
    alpha: float = 0.4,
    blank_canvas: bool = False,
    blur: bool = False,
    overlap: float = 0.0,
    segment: bool = True,
    use_holes: bool = True,
    convert_to_percentiles: bool = True,
    binarize: bool = False,
    thresh: float = 0.5,
    top_left: Optional[Tuple[int, int]] = None,
    bot_right: Optional[Tuple[int, int]] = None,
    custom_downsample: int = 1,
    max_size: Optional[int] = None
) -> Image.Image:
    """
    Create attention heatmap overlay on WSI

    Args:
        scores: Attention scores (N,)
        coords: Patch coordinates at level 0 (N, 2)
        slide_path: Path to WSI file
        wsi_object: Pre-initialized WholeSlideImage object
        vis_level: Pyramid level for visualization (-1 = auto)
        patch_size: Patch size at level 0
        cmap: Matplotlib colormap name
        alpha: Blending factor (0=background only, 1=heatmap only)
        blank_canvas: Use blank canvas instead of original image
        blur: Apply Gaussian blur
        overlap: Patch overlap ratio
        segment: Use tissue segmentation
        use_holes: Mask out holes in tissue
        convert_to_percentiles: Convert scores to percentiles
        binarize: Binarize attention scores
        thresh: Binarization threshold
        top_left: Top-left corner for ROI
        bot_right: Bottom-right corner for ROI
        custom_downsample: Additional downsampling factor
        max_size: Maximum output size

    Returns:
        PIL Image with heatmap overlay
    """
    # Initialize WSI object
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)

    if vis_level < 0:
        vis_level = wsi_object.wsi.get_best_level_for_downsample(32)

    downsample = wsi_object.level_downsamples[vis_level]
    scale = [1/downsample[0], 1/downsample[1]]

    # Flatten scores if needed
    if len(scores.shape) == 2:
        scores = scores.flatten()

    # Set binarization threshold
    if binarize:
        threshold = thresh if thresh >= 0 else 1.0 / len(scores)
    else:
        threshold = 0.0

    # Calculate region size and filter coordinates
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
        coords = coords - top_left
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)
    else:
        region_size = wsi_object.level_dim[vis_level]
        top_left = (0, 0)
        bot_right = wsi_object.level_dim[0]
        w, h = region_size

    # Scale patch size and coordinates
    patch_size_scaled = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords_scaled = np.ceil(coords * np.array(scale)).astype(int)

    print('\nCreating heatmap:')
    print(f'  Region: top_left={top_left}, bot_right={bot_right}')
    print(f'  Size: w={w}, h={h}')
    print(f'  Scaled patch size: {patch_size_scaled}')
    print(f'  Num patches: {len(coords)}')

    # Normalize scores
    if convert_to_percentiles:
        scores = to_percentiles(scores)
    scores = scores / 100.0

    # Create overlay and counter arrays
    overlay = np.zeros(region_size[::-1], dtype=np.float32)
    counter = np.zeros(region_size[::-1], dtype=np.uint16)

    # Accumulate attention scores
    count = 0
    for idx in range(len(coords_scaled)):
        score = scores[idx]
        coord = coords_scaled[idx]

        if score >= threshold:
            if binarize:
                score = 1.0
                count += 1
        else:
            score = 0.0

        # Accumulate
        y_start, y_end = coord[1], coord[1] + patch_size_scaled[1]
        x_start, x_end = coord[0], coord[0] + patch_size_scaled[0]
        overlay[y_start:y_end, x_start:x_end] += score
        counter[y_start:y_end, x_start:x_end] += 1

    if binarize:
        print(f'  Binarized: {count}/{len(coords)} patches positive (threshold={threshold:.4f})')

    # Average overlapping regions
    zero_mask = counter == 0
    if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    del counter

    # Apply Gaussian blur
    if blur:
        kernel_size = tuple((patch_size_scaled * (1 - overlap)).astype(int) * 2 + 1)
        overlay = cv2.GaussianBlur(overlay, kernel_size, 0)

    # Get tissue mask
    if segment:
        tissue_mask = wsi_object.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))

    # Load base image
    if not blank_canvas:
        img = np.array(wsi_object.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
    else:
        img = np.full((region_size[1], region_size[0], 3), 255, dtype=np.uint8)

    # Get colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Apply heatmap overlay
    print('  Applying heatmap overlay...')
    for idx in range(len(coords_scaled)):
        score = scores[idx]
        coord = coords_scaled[idx]

        if score >= threshold:
            y_start, y_end = coord[1], coord[1] + patch_size_scaled[1]
            x_start, x_end = coord[0], coord[0] + patch_size_scaled[0]

            # Get attention block
            raw_block = overlay[y_start:y_end, x_start:x_end]

            # Get image block
            img_block = img[y_start:y_end, x_start:x_end].copy()

            # Apply colormap
            color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

            if segment:
                # Apply tissue mask
                mask_block = tissue_mask[y_start:y_end, x_start:x_end]
                img_block[mask_block] = color_block[mask_block]
            else:
                img_block = color_block

            # Update image
            img[y_start:y_end, x_start:x_end] = img_block

    print('  Done!')
    del overlay

    # Apply final blur
    if blur:
        kernel_size = tuple((patch_size_scaled * (1 - overlap)).astype(int) * 2 + 1)
        img = cv2.GaussianBlur(img, kernel_size, 0)

    # Alpha blending
    if alpha < 1.0 and not blank_canvas:
        print('  Applying alpha blending...')
        original = np.array(wsi_object.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        img = cv2.addWeighted(original, 1 - alpha, img, alpha, 0)

    # Convert to PIL Image
    img = Image.fromarray(img)

    # Resize if needed
    w, h = img.size
    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resize_factor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resize_factor), int(h * resize_factor)))

    return img


def sample_top_patches(
    scores: np.ndarray,
    coords: np.ndarray,
    wsi_object: WholeSlideImage,
    k: int = 15,
    patch_level: int = 0,
    patch_size: int = 256,
    save_dir: Optional[str] = None,
    slide_id: Optional[str] = None
) -> dict:
    """
    Sample top-k patches with highest attention scores

    Args:
        scores: Attention scores (N,)
        coords: Patch coordinates at level 0 (N, 2)
        wsi_object: WholeSlideImage object
        k: Number of top patches to sample
        patch_level: Pyramid level for patch extraction
        patch_size: Patch size
        save_dir: Directory to save patches
        slide_id: Slide identifier for naming

    Returns:
        Dictionary with sampled coordinates and scores
    """
    # Get top-k indices
    top_indices = np.argsort(scores.flatten())[-k:][::-1]
    top_coords = coords[top_indices]
    top_scores = scores.flatten()[top_indices]

    patches = []
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        for idx, (coord, score) in enumerate(zip(top_coords, top_scores)):
            # Extract patch
            patch = wsi_object.wsi.read_region(
                tuple(coord), patch_level, (patch_size, patch_size)
            ).convert('RGB')
            patches.append(patch)

            # Save patch
            if slide_id is not None:
                filename = f'{idx}_{slide_id}_x_{coord[0]}_y_{coord[1]}_score_{score:.4f}.png'
                patch.save(os.path.join(save_dir, filename))

    return {
        'indices': top_indices,
        'coords': top_coords,
        'scores': top_scores,
        'patches': patches
    }


class MILLabVisualizer:
    """
    High-level visualization interface for MIL-Lab models

    Example:
        >>> viz = MILLabVisualizer(model, wsi_path='/path/to/slide.svs')
        >>> viz.segment_tissue()
        >>> heatmap = viz.create_heatmap(
        ...     features=features,
        ...     coords=coords,
        ...     cmap='jet',
        ...     alpha=0.4
        ... )
        >>> heatmap.save('heatmap.png')
    """

    def __init__(self, model, wsi_path: Optional[str] = None):
        """
        Initialize visualizer

        Args:
            model: MIL-Lab model
            wsi_path: Optional path to WSI file
        """
        self.model = model
        self.wsi_object = WholeSlideImage(wsi_path) if wsi_path is not None else None

    def load_wsi(self, wsi_path: str):
        """Load whole slide image"""
        self.wsi_object = WholeSlideImage(wsi_path)

    def segment_tissue(self, **kwargs):
        """Segment tissue regions (see WholeSlideImage.segment_tissue for args)"""
        if self.wsi_object is None:
            raise ValueError("No WSI loaded. Call load_wsi() first.")
        self.wsi_object.segment_tissue(**kwargs)

    def create_heatmap(
        self,
        features: torch.Tensor,
        coords: np.ndarray,
        attention_scores: Optional[np.ndarray] = None,
        **kwargs
    ) -> Image.Image:
        """
        Create attention heatmap

        Args:
            features: Patch features (N, D) or (1, N, D)
            coords: Patch coordinates (N, 2)
            attention_scores: Pre-computed attention scores. If None, will compute from model.
            **kwargs: Additional arguments for draw_heatmap()

        Returns:
            PIL Image with heatmap overlay
        """
        if self.wsi_object is None:
            raise ValueError("No WSI loaded. Call load_wsi() first.")

        # Compute attention scores if not provided
        if attention_scores is None:
            self.model.eval()
            with torch.no_grad():
                if len(features.shape) == 2:
                    features = features.unsqueeze(0)  # Add batch dimension

                results_dict, log_dict = self.model(features.cuda(), return_attention=True)
                attention_scores = log_dict['attention'].cpu().numpy().squeeze()

        # Ensure coords and attention scores have matching lengths
        min_len = min(len(coords), len(attention_scores))
        coords = coords[:min_len]
        attention_scores = attention_scores[:min_len]

        # Create heatmap
        return draw_heatmap(
            scores=attention_scores,
            coords=coords,
            slide_path=None,
            wsi_object=self.wsi_object,
            **kwargs
        )

    def sample_top_patches(
        self,
        attention_scores: np.ndarray,
        coords: np.ndarray,
        **kwargs
    ) -> dict:
        """Sample top-k patches (see sample_top_patches for args)"""
        if self.wsi_object is None:
            raise ValueError("No WSI loaded. Call load_wsi() first.")

        return sample_top_patches(
            scores=attention_scores,
            coords=coords,
            wsi_object=self.wsi_object,
            **kwargs
        )
