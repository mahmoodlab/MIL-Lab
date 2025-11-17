#!/usr/bin/env python3
"""
Visualization utilities for MIL models
Works with both CLAM and ABMIL models
"""

import os
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: openslide not available, patch extraction will not work")

try:
    from src.visualization import TridentVisualizer
    TRIDENT_VIZ_AVAILABLE = True
except ImportError:
    TRIDENT_VIZ_AVAILABLE = False
    print("Warning: TridentVisualizer not available, heatmap visualization will not work")


def load_slide_data(slide_id, feats_path, default_patch_size=256):
    """
    Load features, coordinates, and patch size for a given slide

    Parameters
    ----------
    slide_id : str
        Slide identifier
    feats_path : str
        Path to features directory
    default_patch_size : int
        Default patch size if not in h5 metadata

    Returns
    -------
    patch_features : torch.Tensor
        Features of shape (num_patches, feature_dim)
    coords : np.ndarray
        Coordinates of shape (num_patches, 2)
    patch_size_level0 : int
        Patch size at level 0
    """
    feat_path = os.path.join(feats_path, slide_id + '.h5')

    with h5py.File(feat_path, 'r') as f:
        # Handle both 2D and 3D formats
        raw_features = f['features'][:]
        if len(raw_features.shape) == 3:
            patch_features = torch.from_numpy(raw_features).squeeze(0)
        else:
            patch_features = torch.from_numpy(raw_features)

        # Try to load coordinates
        if 'coords_patching' in f:
            coords = f['coords_patching'][:]
        elif 'coords' in f:
            coords = f['coords'][:]
        else:
            # Create dummy coordinates if not available
            num_patches = patch_features.shape[0]
            coords = np.zeros((num_patches, 2))

        # Try to get patch size from metadata
        if 'coords_patching' in f and hasattr(f['coords_patching'], 'attrs'):
            if 'patch_size' in f['coords_patching'].attrs:
                patch_size_level0 = int(f['coords_patching'].attrs['patch_size'])
            else:
                patch_size_level0 = default_patch_size
        else:
            patch_size_level0 = default_patch_size

    # Ensure coords and features have the same length
    min_len = min(len(coords), len(patch_features))
    coords = coords[:min_len]
    patch_features = patch_features[:min_len]

    return patch_features, coords, patch_size_level0


def get_model_predictions(model, patch_features, label, criterion, device, model_type='abmil'):
    """
    Get model predictions and attention scores

    Parameters
    ----------
    model : torch.nn.Module
        MIL model (CLAM or ABMIL)
    patch_features : torch.Tensor
        Features of shape (num_patches, feature_dim)
    label : int
        True label (needed for CLAM)
    criterion : torch.nn.Module
        Loss function (needed for CLAM)
    device : torch.device
        Device to run on
    model_type : str
        'clam' or 'abmil'

    Returns
    -------
    predicted_class : int
        Predicted class
    attention_scores : np.ndarray
        Attention scores of shape (num_patches,)
    logits : torch.Tensor
        Model logits
    """
    model.eval()
    with torch.no_grad():
        features_input = patch_features.float().to(device).unsqueeze(0)

        if model_type.lower() == 'clam':
            # CLAM requires loss_fn and label
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)
            results_dict, log_dict = model(
                features_input,
                loss_fn=criterion,
                label=label_tensor,
                return_attention=True
            )
        else:
            # ABMIL
            results_dict, log_dict = model(
                features_input,
                return_attention=True
            )

        logits = results_dict['logits']
        attention_scores = log_dict['attention'].cpu().numpy().squeeze()
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class, attention_scores, logits


def extract_top_patches(slide_path, coords, attention_scores, patch_size_level0, top_k=3):
    """
    Extract top K patches based on attention scores

    Parameters
    ----------
    slide_path : str
        Path to whole slide image
    coords : np.ndarray
        Patch coordinates
    attention_scores : np.ndarray
        Attention scores
    patch_size_level0 : int
        Patch size at level 0
    top_k : int
        Number of top patches to extract

    Returns
    -------
    patches : list of PIL.Image
        Top K patches
    scores : list of float
        Attention scores for top K patches
    locations : list of tuple
        (x, y) coordinates for top K patches
    """
    if not OPENSLIDE_AVAILABLE:
        return None, None, None

    # Get indices of top k patches
    top_indices = np.argsort(attention_scores)[-top_k:][::-1]

    # Open slide
    try:
        wsi = openslide.OpenSlide(slide_path)
    except:
        return None, None, None

    patches = []
    scores = []
    locations = []

    for idx in top_indices:
        x, y = coords[idx]
        score = attention_scores[idx]

        # Extract patch at level 0
        patch = wsi.read_region((int(x), int(y)), 0, (patch_size_level0, patch_size_level0))
        patch = patch.convert('RGB')

        patches.append(patch)
        scores.append(score)
        locations.append((x, y))

    wsi.close()
    return patches, scores, locations


def plot_confusion_matrix(
    all_labels,
    all_preds,
    class_labels,
    title='Confusion Matrix',
    output_path='confusion_matrix.png'
):
    """
    Plot and save confusion matrix

    Parameters
    ----------
    all_labels : list
        True labels
    all_preds : list
        Predicted labels
    class_labels : list
        Class label names
    title : str
        Plot title
    output_path : str
        Path to save figure
    """
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def visualize_top_patches(
    model,
    df,
    feats_path,
    wsi_dir,
    num_classes,
    class_labels,
    criterion,
    device,
    model_type='abmil',
    output_dir='./output',
    top_k=3
):
    """
    Visualize top K patches per class

    Parameters
    ----------
    model : torch.nn.Module
        Trained MIL model
    df : pd.DataFrame
        Dataframe with test slides
    feats_path : str
        Path to features directory
    wsi_dir : str
        Path to whole slide images directory
    num_classes : int
        Number of classes
    class_labels : list
        Class label names
    criterion : torch.nn.Module
        Loss function (for CLAM)
    device : torch.device
        Device
    model_type : str
        'clam' or 'abmil'
    output_dir : str
        Output directory
    top_k : int
        Number of top patches to extract
    """
    if not OPENSLIDE_AVAILABLE:
        print("Skipping top patches visualization (openslide not available)")
        return

    # Get one example from each class
    test_df = df[df['split'] == 'test']
    example_slides = []
    for grade in range(num_classes):
        grade_slides = test_df[test_df['label'] == grade]
        if len(grade_slides) > 0:
            example_slides.append(grade_slides.iloc[0])

    print(f"\n{'='*70}")
    print(f"Extracting Top {top_k} Patches Per Class")
    print(f"{'='*70}\n")

    # Create output directory
    top_patches_dir = os.path.join(output_dir, 'top_patches_per_grade')
    os.makedirs(top_patches_dir, exist_ok=True)

    # Create figure
    fig = plt.figure(figsize=(5*top_k, 6*num_classes))
    gs = GridSpec(num_classes, top_k, figure=fig, hspace=0.3, wspace=0.2)

    for grade_idx, slide_row in enumerate(example_slides):
        slide_id = slide_row['slide_id']
        true_label = slide_row['label']

        print(f"Processing {class_labels[true_label]}: {slide_id}")

        # Load slide data
        patch_features, coords, patch_size_level0 = load_slide_data(
            slide_id, feats_path, default_patch_size=256
        )

        # Get predictions and attention
        predicted_class, attention_scores, logits = get_model_predictions(
            model, patch_features, true_label, criterion, device, model_type
        )

        # Extract top K patches
        slide_path = os.path.join(wsi_dir, f'{slide_id}.tiff')
        patches, scores, locations = extract_top_patches(
            slide_path, coords, attention_scores, patch_size_level0, top_k=top_k
        )

        if patches is None:
            print(f"  Could not load slide {slide_id}, skipping...")
            continue

        # Display top K patches
        for patch_idx, (patch, score, (x, y)) in enumerate(zip(patches, scores, locations)):
            ax = fig.add_subplot(gs[grade_idx, patch_idx])
            ax.imshow(patch)
            ax.axis('off')

            title = f'Score: {score:.4f}\n(x={int(x)}, y={int(y)})'
            ax.set_title(title, fontsize=9)

            # Save individual patch
            patch_filename = f'Class{true_label}_{slide_id[:8]}_top{patch_idx+1}_score{score:.4f}.png'
            patch.save(os.path.join(top_patches_dir, patch_filename))

        # Add row label
        spacing = 0.92 - (grade_idx * (0.80 / num_classes))
        fig.text(0.08, spacing, class_labels[true_label],
                 va='center', ha='right', fontsize=14, fontweight='bold')

        print(f"  Top {top_k} attention scores: {scores}")
        print(f"  Predicted: {class_labels[predicted_class]}\n")

    # Add column headers
    for i in range(top_k):
        fig.text(0.30 + i*0.20, 0.98, f'Top {i+1}', ha='center', fontsize=14, fontweight='bold')

    plt.suptitle(f'Top {top_k} Patches by Attention Score ({model_type.upper()})',
                 fontsize=16, fontweight='bold', y=0.995)

    output_path = os.path.join(output_dir, f'top_{top_k}_patches_per_grade.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*70}")
    print("Top Patches Visualization Complete!")
    print(f"{'='*70}")
    print(f"Saved to: {output_path}")
    print(f"Individual patches saved to: {top_patches_dir}/")


def visualize_heatmaps(
    model,
    df,
    feats_path,
    wsi_dir,
    num_classes,
    class_labels,
    criterion,
    device,
    model_type='abmil',
    output_dir='./output'
):
    """
    Visualize attention heatmaps overlaid on WSIs

    Parameters
    ----------
    model : torch.nn.Module
        Trained MIL model
    df : pd.DataFrame
        Dataframe with test slides
    feats_path : str
        Path to features directory
    wsi_dir : str
        Path to whole slide images directory
    num_classes : int
        Number of classes
    class_labels : list
        Class label names
    criterion : torch.nn.Module
        Loss function (for CLAM)
    device : torch.device
        Device
    model_type : str
        'clam' or 'abmil'
    output_dir : str
        Output directory
    """
    if not TRIDENT_VIZ_AVAILABLE:
        print("Skipping heatmap visualization (TridentVisualizer not available)")
        return

    # Get one example from each class
    test_df = df[df['split'] == 'test']
    example_slides = []
    for grade in range(num_classes):
        grade_slides = test_df[test_df['label'] == grade]
        if len(grade_slides) > 0:
            example_slides.append(grade_slides.iloc[0])

    print(f"\n{'='*70}")
    print(f"{model_type.upper()} Attention Heatmap Visualization")
    print(f"{'='*70}")
    print(f"Visualizing {len(example_slides)} slides from different classes\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    if num_classes == 3:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    elif num_classes == 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, slide_row in enumerate(example_slides):
        slide_id = slide_row['slide_id']
        true_label = slide_row['label']

        print(f"Processing slide {idx+1}/{len(example_slides)}: {slide_id} ({class_labels[true_label]})")

        # Load slide data
        patch_features, coords, patch_size_level0 = load_slide_data(
            slide_id, feats_path, default_patch_size=256
        )

        # Get predictions and attention
        predicted_class, attention_scores, logits = get_model_predictions(
            model, patch_features, true_label, criterion, device, model_type
        )

        # Create heatmap
        slide_path = os.path.join(wsi_dir, f'{slide_id}.tiff')
        viz = TridentVisualizer(model, wsi_path=slide_path)

        output_filename = f'{slide_id}_Class{true_label}_pred{predicted_class}_{model_type}.png'

        heatmap = viz.create_heatmap(
            features=patch_features,
            coords=coords,
            attention_scores=attention_scores,
            patch_size_level0=patch_size_level0,
            vis_level=-1,
            cmap='jet',
            alpha=0.4,
            normalize=True,
            output_path=os.path.join(output_dir, output_filename)
        )

        # Display heatmap with colorbar
        im = axes[idx].imshow(heatmap)
        axes[idx].axis('off')

        # Add colorbar
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='jet'), cax=cax)
        cbar.set_label('Attention Score', rotation=270, labelpad=15)

        # Title
        title_color = 'green' if predicted_class == true_label else 'red'
        axes[idx].set_title(
            f'{class_labels[true_label]} → Pred: {class_labels[predicted_class]}\nSlide: {slide_id}',
            fontsize=12, fontweight='bold', color=title_color, pad=10
        )

        print(f"  Attention range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")
        print(f"  Predicted: {class_labels[predicted_class]}\n")

    # Hide unused subplots
    total_subplots = 3 if num_classes == 3 else (4 if num_classes == 4 else 6)
    for idx in range(len(example_slides), total_subplots):
        axes[idx].axis('off')

    plt.suptitle(f'{model_type.upper()} Attention Heatmaps - One Example per Class',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    comparison_path = os.path.join(output_dir, f'all_grades_comparison_{model_type}.png')
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*70}")
    print("Heatmap Visualization Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Comparison grid: {comparison_path}")
