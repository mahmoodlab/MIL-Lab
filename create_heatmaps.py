"""
Heatmap Generation Script for MIL-Lab Models

This script generates attention heatmaps for whole slide images using trained MIL models.
The output format is compatible with OpenSlide for synchronized viewing.

Usage:
    python create_heatmaps.py --config heatmaps/configs/config.yaml

Inspired by CLAM (https://github.com/mahmoodlab/CLAM)
Adapted for MIL-Lab framework
"""

import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.visualization import WholeSlideImage, draw_heatmap, sample_top_patches
from src.builder import create_model
from utils.file_utils import save_hdf5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(config):
    """Load model from checkpoint using MIL-Lab builder"""
    model_config = config['model_arguments']

    # Get checkpoint path
    ckpt_path = model_config['ckpt_path']
    print(f'Loading checkpoint from: {ckpt_path}')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model name from config
    # Format: {model_type}.{size}.{encoder}.{postfix}
    model_type = model_config.get('model_type', 'abmil')
    model_size = model_config.get('model_size', 'base')
    encoder = model_config.get('encoder', 'uni')
    postfix = model_config.get('postfix', '')

    if postfix:
        model_name = f"{model_type}.{model_size}.{encoder}.{postfix}"
    else:
        model_name = f"{model_type}.{model_size}.{encoder}"

    # Create model using MIL-Lab's create_model
    try:
        model = create_model(
            model_name=model_name,
            num_classes=model_config.get('n_classes', 2),
            checkpoint_path=ckpt_path,
            **{k: v for k, v in model_config.items()
               if k not in ['ckpt_path', 'model_type', 'model_size', 'encoder', 'postfix', 'n_classes']}
        )
    except Exception as e:
        # Fallback: try loading checkpoint directly
        print(f"Warning: Could not use create_model: {e}")
        print("Attempting direct checkpoint loading...")

        # Try to infer model from checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Import model classes
        from src.models.abmil import ABMIL
        from src.models.clam import CLAM_MB, CLAM_SB

        # Determine model type from config or checkpoint
        if 'clam_mb' in model_type.lower():
            model = CLAM_MB(
                n_classes=model_config.get('n_classes', 2),
                size_arg=model_config.get('model_size', 'small'),
                dropout=model_config.get('dropout', 0.25),
                in_dim=model_config.get('in_dim', 1024),
            )
        elif 'clam_sb' in model_type.lower() or 'clam' in model_type.lower():
            model = CLAM_SB(
                n_classes=model_config.get('n_classes', 2),
                size_arg=model_config.get('model_size', 'small'),
                dropout=model_config.get('dropout', 0.25),
                in_dim=model_config.get('in_dim', 1024),
            )
        elif 'abmil' in model_type.lower():
            model = ABMIL(
                in_dim=model_config.get('in_dim', 1024),
                n_classes=model_config.get('n_classes', 2),
                dropout=model_config.get('dropout', 0.25),
                act=model_config.get('act', 'relu'),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print('Model loaded successfully!')
    return model


def infer_single_slide(model, features, label=None, return_attention=True):
    """
    Run inference on a single slide

    Args:
        model: MIL model
        features: Patch features (N, D)
        label: Ground truth label (optional)
        return_attention: Whether to return attention scores

    Returns:
        Dictionary with predictions and attention scores
    """
    features = features.to(device)

    with torch.inference_mode():
        # Add batch dimension if needed
        if len(features.shape) == 2:
            features = features.unsqueeze(0)

        # Get model predictions
        results_dict, log_dict = model(features, return_attention=return_attention)

        # Extract results
        logits = results_dict.get('logits', None)
        Y_prob = results_dict.get('Y_prob', None)
        Y_hat = results_dict.get('Y_hat', None)

        if Y_hat is not None:
            Y_hat = Y_hat.item()

        # Get attention scores
        if return_attention and 'attention' in log_dict:
            A = log_dict['attention'].cpu().numpy().squeeze()
        else:
            A = None

        # Print results
        if Y_prob is not None:
            print(f'Y_hat: {Y_hat}, Y_prob: {[f"{p:.4f}" for p in Y_prob.cpu().flatten()]}')
            if label is not None:
                print(f'Ground truth label: {label}')

        return {
            'Y_hat': Y_hat,
            'Y_prob': Y_prob.cpu().numpy() if Y_prob is not None else None,
            'logits': logits.cpu().numpy() if logits is not None else None,
            'attention': A
        }


def process_single_slide(slide_info, config, model, wsi_object=None):
    """
    Process a single slide and generate heatmaps

    Args:
        slide_info: Dictionary with slide information
        config: Configuration dictionary
        model: MIL model
        wsi_object: Pre-initialized WholeSlideImage object (optional)

    Returns:
        Updated slide_info with results
    """
    data_args = config['data_arguments']
    exp_args = config['exp_arguments']
    heatmap_args = config['heatmap_arguments']
    patch_args = config['patching_arguments']

    slide_id = slide_info['slide_id']
    slide_ext = data_args['slide_ext']

    if slide_ext not in slide_id:
        slide_name = slide_id + slide_ext
    else:
        slide_name = slide_id
        slide_id = slide_id.replace(slide_ext, '')

    print(f'\n{"="*80}')
    print(f'Processing: {slide_name}')
    print(f'{"="*80}')

    # Get label
    label = slide_info.get('label', 'Unspecified')

    # Determine slide path
    if isinstance(data_args['data_dir'], str):
        slide_path = os.path.join(data_args['data_dir'], slide_name)
    elif isinstance(data_args['data_dir'], dict):
        data_dir_key = slide_info.get(data_args.get('data_dir_key', 'source'))
        slide_path = os.path.join(data_args['data_dir'][data_dir_key], slide_name)
    else:
        raise ValueError("data_dir must be str or dict")

    if not os.path.exists(slide_path):
        print(f'ERROR: Slide not found at {slide_path}')
        return slide_info

    # Create output directories
    raw_save_dir = os.path.join(
        exp_args['raw_save_dir'],
        exp_args['save_exp_code'],
        str(label),
        slide_id
    )
    os.makedirs(raw_save_dir, exist_ok=True)

    production_save_dir = os.path.join(
        exp_args['production_save_dir'],
        exp_args['save_exp_code'],
        str(label)
    )
    os.makedirs(production_save_dir, exist_ok=True)

    # Load features
    features_path = slide_info.get('features_path')
    if features_path is None:
        # Try default location
        features_path = os.path.join(data_args.get('features_dir', 'features'), slide_id + '.pt')

    if not os.path.exists(features_path):
        print(f'ERROR: Features not found at {features_path}')
        return slide_info

    print(f'Loading features from: {features_path}')
    features = torch.load(features_path)

    # Load coordinates
    coords_path = slide_info.get('coords_path')
    if coords_path is None:
        # Try h5 file
        h5_path = features_path.replace('.pt', '.h5')
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                coords = f['coords'][:]
        else:
            print(f'ERROR: Coordinates not found')
            return slide_info
    else:
        with h5py.File(coords_path, 'r') as f:
            coords = f['coords'][:]

    print(f'Loaded {len(features)} features and {len(coords)} coordinates')
    slide_info['bag_size'] = len(features)

    # Run inference
    print('Running inference...')
    results = infer_single_slide(model, features, label=label, return_attention=True)

    Y_hat = results['Y_hat']
    Y_prob = results['Y_prob']
    A = results['attention']

    if A is None:
        print('ERROR: No attention scores returned from model')
        return slide_info

    # Ensure attention scores have correct shape
    if len(A.shape) > 1:
        A = A.flatten()

    # Match lengths
    min_len = min(len(A), len(coords))
    A = A[:min_len]
    coords = coords[:min_len]

    print(f'Attention scores shape: {A.shape}')
    print(f'Coordinates shape: {coords.shape}')
    print(f'Attention range: [{A.min():.4f}, {A.max():.4f}]')

    # Save attention scores and coordinates
    block_map_save_path = os.path.join(raw_save_dir, f'{slide_id}_blockmap.h5')
    asset_dict = {'attention_scores': A.reshape(-1, 1), 'coords': coords}
    save_hdf5(block_map_save_path, asset_dict, mode='w')
    print(f'Saved attention scores to: {block_map_save_path}')

    # Store predictions
    if Y_prob is not None:
        for c in range(len(Y_prob)):
            slide_info[f'Pred_{c}'] = Y_hat if c == Y_hat else None
            slide_info[f'p_{c}'] = Y_prob[c]

    # Initialize WSI for heatmap generation
    if wsi_object is None:
        print('Initializing WSI object...')
        wsi_object = WholeSlideImage(slide_path)

        # Segment tissue
        seg_params = {
            'seg_level': -1,
            'sthresh': 15,
            'mthresh': 11,
            'close': 2,
            'use_otsu': False
        }
        filter_params = {
            'a_t': 50.0,
            'a_h': 8.0,
            'max_n_holes': 10
        }

        wsi_object.segment_tissue(**seg_params, filter_params=filter_params)
        print('Tissue segmentation complete!')

    # Generate heatmap
    print('Generating heatmap...')

    patch_size = (patch_args['patch_size'], patch_args['patch_size'])
    vis_level = heatmap_args.get('vis_level', -1)

    heatmap = draw_heatmap(
        scores=A,
        coords=coords,
        slide_path=slide_path,
        wsi_object=wsi_object,
        vis_level=vis_level,
        patch_size=patch_size,
        cmap=heatmap_args.get('cmap', 'jet'),
        alpha=heatmap_args.get('alpha', 0.4),
        blank_canvas=heatmap_args.get('blank_canvas', False),
        blur=heatmap_args.get('blur', False),
        overlap=patch_args.get('overlap', 0.0),
        segment=True,
        use_holes=True,
        convert_to_percentiles=heatmap_args.get('convert_to_percentiles', True),
        binarize=heatmap_args.get('binarize', False),
        thresh=heatmap_args.get('binary_thresh', -1),
        custom_downsample=heatmap_args.get('custom_downsample', 1)
    )

    # Save heatmap
    heatmap_filename = f'{slide_id}_heatmap.{heatmap_args.get("save_ext", "png")}'
    heatmap_path = os.path.join(production_save_dir, heatmap_filename)

    if heatmap_args.get('save_ext', 'png') == 'jpg':
        heatmap.save(heatmap_path, quality=100)
    else:
        heatmap.save(heatmap_path)

    print(f'Saved heatmap to: {heatmap_path}')

    # Save original image if requested
    if heatmap_args.get('save_orig', False):
        if vis_level < 0:
            vis_level = wsi_object.wsi.get_best_level_for_downsample(32)

        orig_img = wsi_object.wsi.read_region(
            (0, 0), vis_level, wsi_object.level_dim[vis_level]
        ).convert('RGB')

        orig_filename = f'{slide_id}_original.{heatmap_args.get("save_ext", "png")}'
        orig_path = os.path.join(production_save_dir, orig_filename)
        orig_img.save(orig_path)
        print(f'Saved original image to: {orig_path}')

    # Sample top patches if requested
    if 'sample_arguments' in config:
        for sample_config in config['sample_arguments'].get('samples', []):
            if sample_config.get('sample', False):
                print(f"\nSampling patches: {sample_config['name']}")

                sample_dir = os.path.join(
                    production_save_dir,
                    'sampled_patches',
                    f'label_{label}_pred_{Y_hat}',
                    sample_config['name']
                )

                sample_results = sample_top_patches(
                    scores=A,
                    coords=coords,
                    wsi_object=wsi_object,
                    k=sample_config.get('k', 15),
                    patch_level=patch_args.get('patch_level', 0),
                    patch_size=patch_args['patch_size'],
                    save_dir=sample_dir,
                    slide_id=slide_id
                )

                print(f'Saved {len(sample_results["coords"])} patches to: {sample_dir}')

    return slide_info


def main():
    parser = argparse.ArgumentParser(description='Generate attention heatmaps for MIL models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--save_exp_code', type=str, default=None,
                       help='Experiment code (overrides config)')
    args = parser.parse_args()

    # Load configuration
    print(f'Loading configuration from: {args.config}')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override experiment code if provided
    if args.save_exp_code is not None:
        config['exp_arguments']['save_exp_code'] = args.save_exp_code

    # Print configuration
    print('\n' + '='*80)
    print('CONFIGURATION')
    print('='*80)
    for section, params in config.items():
        if isinstance(params, dict):
            print(f'\n{section}:')
            for key, value in params.items():
                print(f'  {key}: {value}')
        else:
            print(f'{section}: {params}')
    print('='*80 + '\n')

    # Confirm
    decision = input('Continue? [Y/n] ')
    if decision.lower() in ['n', 'no']:
        print('Aborted.')
        return

    # Load slide list
    data_args = config['data_arguments']

    if data_args.get('process_list') is not None:
        # Load from CSV
        process_list_path = data_args['process_list']
        if not os.path.isabs(process_list_path):
            process_list_path = os.path.join('heatmaps/process_lists', process_list_path)

        print(f'Loading process list from: {process_list_path}')
        df = pd.read_csv(process_list_path)
    else:
        # Create from directory
        if isinstance(data_args['data_dir'], list):
            slides = []
            for data_dir in data_args['data_dir']:
                slides.extend([f for f in os.listdir(data_dir)
                             if data_args['slide_ext'] in f])
        else:
            slides = [f for f in os.listdir(data_args['data_dir'])
                     if data_args['slide_ext'] in f]

        df = pd.DataFrame({'slide_id': slides, 'process': 1})

    # Filter slides to process
    if 'process' not in df.columns:
        df['process'] = 1

    process_stack = df[df['process'] == 1].reset_index(drop=True)
    total = len(process_stack)

    print(f'\nFound {total} slides to process:')
    print(process_stack[['slide_id']].head(min(10, total)))
    if total > 10:
        print(f'... and {total - 10} more')

    # Load model
    print('\n' + '='*80)
    print('LOADING MODEL')
    print('='*80)
    model = load_model_from_checkpoint(config)

    # Process slides
    print('\n' + '='*80)
    print('PROCESSING SLIDES')
    print('='*80)

    results = []
    for idx in tqdm(range(len(process_stack)), desc='Processing slides'):
        slide_info = process_stack.loc[idx].to_dict()

        try:
            result = process_single_slide(slide_info, config, model)
            results.append(result)
        except Exception as e:
            print(f'\nERROR processing {slide_info["slide_id"]}: {str(e)}')
            import traceback
            traceback.print_exc()
            results.append(slide_info)

    # Save results
    results_df = pd.DataFrame(results)
    results_dir = os.path.join('heatmaps', 'results')
    os.makedirs(results_dir, exist_ok=True)

    exp_code = config['exp_arguments']['save_exp_code']
    results_path = os.path.join(results_dir, f'{exp_code}_results.csv')
    results_df.to_csv(results_path, index=False)

    print(f'\n{"="*80}')
    print('COMPLETE')
    print(f'{"="*80}')
    print(f'Results saved to: {results_path}')
    print(f'Heatmaps saved to: {config["exp_arguments"]["production_save_dir"]}')
    print(f'Raw data saved to: {config["exp_arguments"]["raw_save_dir"]}')

    # Save config
    config_save_path = os.path.join(
        config['exp_arguments']['raw_save_dir'],
        exp_code,
        'config.yaml'
    )
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f'Config saved to: {config_save_path}')


if __name__ == '__main__':
    main()
