#!/usr/bin/env python3
"""
High-Resolution Heatmap Generation with Overlapping Patches

This script generates dense, high-resolution heatmaps by:
1. Creating a dense grid of overlapping patches (e.g., 90-95% overlap)
2. Extracting features for each patch on-the-fly
3. Getting attention scores from the model
4. Creating a high-resolution heatmap at 20x (or any resolution)

Usage:
    python generate_heatmaps_highres.py \
        --checkpoint path/to/model.pt \
        --slide path/to/slide.tiff \
        --output heatmap_highres.tiff \
        --overlap 0.90 \
        --vis_level 0
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import h5py

from src.visualization import WholeSlideImage, draw_heatmap
from src.builder import create_model
from utils.file_utils import save_hdf5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect_feature_dim_from_checkpoint(checkpoint_path):
    """
    Detect the feature dimension expected by the model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        in_dim: Feature dimension (e.g., 1024, 1536, 768, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Look for patch_embed layer which has shape (out_features, in_features)
    if 'model.patch_embed.0.weight' in checkpoint:
        weight_shape = checkpoint['model.patch_embed.0.weight'].shape
        in_dim = weight_shape[1]  # in_features
        print(f"  Detected feature dimension from checkpoint: {in_dim}")
        return in_dim
    else:
        raise ValueError(
            f"Could not detect feature dimension from checkpoint. "
            f"Expected 'model.patch_embed.0.weight' but found keys: {list(checkpoint.keys())[:10]}"
        )


# Mapping from feature dimension to recommended feature extractor
FEATURE_DIM_TO_EXTRACTOR = {
    1024: 'uni_v1',      # UNI v1 or ResNet50
    1536: 'uni_v2',      # UNI v2 or GigaPath or H-Optimus
    768: 'ctranspath',   # CTransPath or Phikon or CONCH v1.5
    512: 'conch_v1',     # CONCH v1
    2560: 'virchow',     # Virchow/Virchow2
}


def create_dense_patch_grid(wsi, patch_size=256, step_size=26, level=0, top_left=None, bot_right=None):
    """
    Create a dense grid of patch coordinates with overlapping patches

    Args:
        wsi: WholeSlideImage object
        patch_size: Size of each patch
        step_size: Step size between patches (smaller = more overlap)
        level: Pyramid level
        top_left: Optional ROI top-left corner
        bot_right: Optional ROI bottom-right corner

    Returns:
        coords: (N, 2) array of patch coordinates at level 0
    """
    # Get dimensions at level 0
    if top_left is None or bot_right is None:
        w, h = wsi.level_dim[0]
        top_left = (0, 0)
        bot_right = (w, h)
    else:
        w = bot_right[0] - top_left[0]
        h = bot_right[1] - top_left[1]

    # Create grid
    x_coords = np.arange(top_left[0], bot_right[0] - patch_size + 1, step_size)
    y_coords = np.arange(top_left[1], bot_right[1] - patch_size + 1, step_size)

    # Create meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.int32)

    print(f"  Created dense grid: {len(coords)} patches")
    print(f"  Grid dimensions: {len(x_coords)} x {len(y_coords)}")
    print(f"  Step size: {step_size}px (overlap: {(1 - step_size/patch_size)*100:.1f}%)")

    return coords


def load_feature_extractor(model_name='uni_v2', feature_dim=None):
    """
    Load feature extractor model

    Args:
        model_name: Feature extractor name ('uni_v1', 'uni_v2', 'resnet50', 'ctranspath', etc.)
        feature_dim: Expected feature dimension (used for validation)

    Returns:
        model: Feature extractor model
        transform: Image transform
    """
    from torchvision import transforms
    import timm

    if model_name == 'uni_v2':
        # UNI v2 (UNI2-h) - 1536-dimensional features
        try:
            timm_kwargs = {
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667 * 2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True
            }

            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h",
                pretrained=True,
                **timm_kwargs
            )
            model.eval()

            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            print(f"  ✓ Loaded UNI v2 (UNI2-h) feature extractor (1536-dim)")
            return model, transform

        except Exception as e:
            print(f"  ✗ Error loading UNI v2: {e}")
            raise

    elif model_name == 'uni_v1':
        # UNI v1 - 1024-dimensional features
        try:
            timm_kwargs = {
                'img_size': 224,
                'patch_size': 16,
                'init_values': 1e-5,
                'num_classes': 0,
                'dynamic_img_size': True,
            }

            model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                **timm_kwargs
            )
            model.eval()

            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            print(f"  ✓ Loaded UNI v1 feature extractor (1024-dim)")
            return model, transform

        except Exception as e:
            print(f"  ✗ Error loading UNI v1: {e}")
            raise

    elif model_name == 'resnet50':
        # ResNet50 - 1024-dimensional features
        model = timm.create_model('resnet50', pretrained=True, num_classes=0)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"  ✓ Loaded ResNet50 feature extractor (1024-dim)")
        return model, transform

    elif model_name == 'ctranspath':
        # CTransPath - 768-dimensional features
        print(f"  Note: CTransPath requires special setup. Using as fallback.")
        print(f"  If you need CTransPath, please use Trident's feature extraction instead.")
        raise NotImplementedError("CTransPath not yet implemented in this script")

    elif model_name == 'conch_v1':
        # CONCH v1 - 512-dimensional features
        print(f"  Note: CONCH v1 requires special setup. Using as fallback.")
        print(f"  If you need CONCH v1, please use Trident's feature extraction instead.")
        raise NotImplementedError("CONCH v1 not yet implemented in this script")

    elif model_name == 'virchow':
        # Virchow - 2560-dimensional features
        print(f"  Note: Virchow requires special setup. Using as fallback.")
        print(f"  If you need Virchow, please use Trident's feature extraction instead.")
        raise NotImplementedError("Virchow not yet implemented in this script")

    else:
        raise ValueError(
            f"Unknown feature extractor: {model_name}. "
            f"Supported: uni_v1, uni_v2, resnet50. "
            f"For other extractors (ctranspath, conch, virchow), use Trident for feature extraction first."
        )


def extract_patch_features(wsi, coords, patch_size, feature_extractor, transform, batch_size=32, level=0):
    """
    Extract features for patches at given coordinates

    Args:
        wsi: WholeSlideImage object
        coords: (N, 2) array of coordinates
        patch_size: Patch size
        feature_extractor: Feature extraction model
        transform: Image transform
        batch_size: Batch size for extraction
        level: Pyramid level for extraction

    Returns:
        features: (N, D) tensor of features
    """
    all_features = []

    print(f"  Extracting features for {len(coords)} patches...")
    for i in tqdm(range(0, len(coords), batch_size), desc="  Batches"):
        batch_coords = coords[i:i+batch_size]
        batch_patches = []

        # Read patches
        for coord in batch_coords:
            patch = wsi.wsi.read_region(
                tuple(coord), level, (patch_size, patch_size)
            ).convert('RGB')
            patch_tensor = transform(patch)
            batch_patches.append(patch_tensor)

        # Stack and extract features
        batch_tensor = torch.stack(batch_patches).to(device)

        with torch.inference_mode():
            features = feature_extractor(batch_tensor)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def get_dense_attention_scores(model, features, batch_size=512):
    """
    Get attention scores for all features by treating them as one large bag

    Args:
        model: MIL model
        features: (N, D) tensor of features
        batch_size: Not used - all features processed together as one bag

    Returns:
        attention: (N,) array of attention scores
    """
    model.eval()

    print(f"  Computing attention for {len(features)} patches...")
    print(f"    Processing all patches as one bag...")

    # Process ALL features as a single bag (required for MIL attention)
    features_batch = features.unsqueeze(0).to(device)  # (1, N, D)

    print(f"    Input shape: {features_batch.shape}")

    with torch.inference_mode():
        # Get attention for entire bag
        _, log_dict = model(features_batch, return_attention=True)

        if 'attention' not in log_dict or log_dict['attention'] is None:
            raise ValueError("Model did not return attention scores")

        attention_raw = log_dict['attention'].cpu()
        print(f"    Raw attention shape: {attention_raw.shape}")

        attention_softmax = torch.softmax(attention_raw, dim=-1)
        attention = attention_softmax.numpy().squeeze()

    print(f"    Final attention shape: {attention.shape}")
    return attention


def generate_highres_heatmap(
    slide_path,
    model,
    feature_extractor,
    transform,
    output_path,
    overlap=0.90,
    patch_size=256,
    vis_level=0,
    batch_size_extract=32,
    batch_size_infer=512,
    cmap='jet',
    alpha=0.5,
    save_coords=True
):
    """
    Generate high-resolution heatmap with dense overlapping patches

    Args:
        slide_path: Path to WSI
        model: Trained MIL model
        feature_extractor: Feature extraction model
        output_path: Output path for heatmap
        overlap: Overlap ratio (0.9 = 90% overlap, step_size = 10% of patch_size)
        patch_size: Patch size
        vis_level: Visualization level (0 = full resolution)
        batch_size_extract: Batch size for feature extraction
        batch_size_infer: Batch size for attention inference
        cmap: Colormap
        alpha: Transparency
        save_coords: Whether to save coordinates and attention scores
    """
    print(f"\n{'='*70}")
    print(f"HIGH-RESOLUTION HEATMAP GENERATION")
    print(f"{'='*70}")
    print(f"Slide: {Path(slide_path).name}")
    print(f"Overlap: {overlap*100:.0f}%")
    print(f"Patch size: {patch_size}px")
    print(f"Visualization level: {vis_level}")

    # Calculate step size
    step_size = int(patch_size * (1 - overlap))
    print(f"Step size: {step_size}px")

    # Load slide
    print(f"\n1. Loading slide...")
    wsi = WholeSlideImage(slide_path)
    print(f"   Dimensions: {wsi.level_dim[0]}")
    print(f"   Levels: {len(wsi.level_dim)}")

    # Create dense grid
    print(f"\n2. Creating dense patch grid...")
    coords = create_dense_patch_grid(
        wsi,
        patch_size=patch_size,
        step_size=step_size,
        level=0
    )

    # Extract features
    print(f"\n3. Extracting features...")
    features = extract_patch_features(
        wsi, coords, patch_size, feature_extractor, transform,
        batch_size=batch_size_extract, level=0
    )
    print(f"   Features shape: {features.shape}")

    # Get attention scores
    print(f"\n4. Computing attention scores...")
    attention = get_dense_attention_scores(
        model, features, batch_size=batch_size_infer
    )
    print(f"   Attention shape: {attention.shape}")
    print(f"   Attention range: [{attention.min():.6f}, {attention.max():.6f}]")
    print(f"   Attention mean: {attention.mean():.6f}")

    # Save coordinates and attention if requested
    if save_coords:
        coords_path = output_path.replace('.tiff', '_coords.h5').replace('.png', '_coords.h5')
        print(f"\n5. Saving coordinates and attention...")
        asset_dict = {
            'coords': coords,
            'attention_scores': attention.reshape(-1, 1)
        }
        save_hdf5(coords_path, asset_dict, mode='w')
        print(f"   Saved to: {coords_path}")

    # Generate heatmap
    print(f"\n6. Generating heatmap...")
    heatmap = draw_heatmap(
        scores=attention,
        coords=coords,
        slide_path=slide_path,
        wsi_object=wsi,
        vis_level=vis_level,
        patch_size=(patch_size, patch_size),
        cmap=cmap,
        alpha=alpha,
        convert_to_percentiles=True,
        segment=False,
        blank_canvas=False,
        blur=False,
        overlap=overlap  # Pass overlap for potential blur radius
    )

    # Save
    print(f"\n7. Saving heatmap...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if output_path.endswith('.tiff'):
        # Save as TIFF with compression
        heatmap.save(output_path, compression='tiff_lzw')
    else:
        heatmap.save(output_path)

    print(f"   Saved to: {output_path}")
    print(f"   Size: {heatmap.size}")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate high-resolution heatmaps')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--slide', type=str, required=True,
                       help='Path to WSI file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output heatmap path (.tiff or .png)')

    # Model config
    parser.add_argument('--model_name', type=str, default='abmil.base.uni_v2.pc108-24k',
                       help='Model name')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--feature_extractor', type=str, default='auto',
                       help='Feature extractor name: auto (detect from checkpoint), uni_v1, uni_v2, resnet50 (default: auto)')

    # Dense sampling
    parser.add_argument('--overlap', type=float, default=0.90,
                       help='Overlap ratio: 0.9=90%%, 0.95=95%% (default: 0.90)')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size (default: 256)')

    # Visualization
    parser.add_argument('--vis_level', type=int, default=0,
                       help='Visualization level: 0=full res, 1=2x downsample, etc. (default: 0)')
    parser.add_argument('--cmap', type=str, default='jet',
                       help='Colormap (default: jet)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency (default: 0.5)')

    # Performance
    parser.add_argument('--batch_size_extract', type=int, default=32,
                       help='Batch size for feature extraction (default: 32)')
    parser.add_argument('--batch_size_infer', type=int, default=512,
                       help='Batch size for attention inference (default: 512)')

    # Options
    parser.add_argument('--no_save_coords', action='store_true',
                       help='Do not save coordinates and attention scores')

    args = parser.parse_args()

    # Load models
    print("Loading models...")
    print()

    # Auto-detect feature extractor if requested
    if args.feature_extractor == 'auto':
        print("Auto-detecting feature extractor from checkpoint...")
        feature_dim = detect_feature_dim_from_checkpoint(args.checkpoint)

        if feature_dim in FEATURE_DIM_TO_EXTRACTOR:
            extractor_name = FEATURE_DIM_TO_EXTRACTOR[feature_dim]
            print(f"  → Recommended extractor for {feature_dim}-dim features: {extractor_name}")
        else:
            print(f"  ⚠ Warning: Unusual feature dimension {feature_dim}")
            print(f"  → Falling back to uni_v2 (may not match!)")
            extractor_name = 'uni_v2'
    else:
        extractor_name = args.feature_extractor
        print(f"Using specified feature extractor: {extractor_name}")

    # Load MIL model
    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint
    ).to(device)
    print(f"✓ MIL model loaded: {model.__class__.__name__}")
    print()

    # Load feature extractor
    feature_extractor, transform = load_feature_extractor(extractor_name)
    feature_extractor = feature_extractor.to(device)
    print()

    # Generate high-res heatmap
    generate_highres_heatmap(
        slide_path=args.slide,
        model=model,
        feature_extractor=feature_extractor,
        transform=transform,
        output_path=args.output,
        overlap=args.overlap,
        patch_size=args.patch_size,
        vis_level=args.vis_level,
        batch_size_extract=args.batch_size_extract,
        batch_size_infer=args.batch_size_infer,
        cmap=args.cmap,
        alpha=args.alpha,
        save_coords=not args.no_save_coords
    )


if __name__ == '__main__':
    main()
