#!/usr/bin/env python3
"""
Standalone visualization script for MIL models
Works with both CLAM and ABMIL models
"""

import argparse
import torch
import torch.nn as nn

from utils import preprocess_panda_data, visualize_top_patches, visualize_heatmaps
from src.builder import create_model

def main():
    parser = argparse.ArgumentParser(description='Visualize MIL model predictions')
    parser.add_argument('--model-type', type=str, required=True, choices=['clam', 'abmil'],
                        help='Type of model (clam or abmil)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--csv-path', type=str, required=True,
                        help='Path to PANDA train.csv')
    parser.add_argument('--feats-path', type=str, required=True,
                        help='Path to features directory')
    parser.add_argument('--wsi-dir', type=str, required=True,
                        help='Path to whole slide images directory')
    parser.add_argument('--output-dir', type=str, default='./visualization_output',
                        help='Output directory for visualizations')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--grade-group', action='store_true',
                        help='Use grade grouping')
    parser.add_argument('--exclude-mid-grade', action='store_true',
                        help='Exclude mid grade (requires --grade-group)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top patches to visualize')
    parser.add_argument('--skip-patches', action='store_true',
                        help='Skip top patches visualization')
    parser.add_argument('--skip-heatmaps', action='store_true',
                        help='Skip heatmap visualization')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed')

    args = parser.parse_args()

    # Preprocess data
    df, num_classes, class_labels = preprocess_panda_data(
        args.csv_path,
        args.feats_path,
        grade_group=args.grade_group,
        exclude_mid_grade=args.exclude_mid_grade,
        seed=args.seed
    )

    # Verify num_classes matches
    if num_classes != args.num_classes:
        raise ValueError(f"num_classes mismatch: expected {args.num_classes}, got {num_classes}")

    # Load model
    print("\n" + "="*70)
    print(f"LOADING {args.model_type.upper()} MODEL")
    print("="*70 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'clam':
        model = create_model(
            'clam.base_ce.uni_v2.none',
            num_classes=num_classes,
            k_sample=8,
            bag_weight=0.7
        ).to(device)
    else:  # abmil
        model = create_model(
            'abmil.base.uni_v2.pc108-24k',
            num_classes=num_classes,
            dropout=0.2,
            gate=True
        ).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print(f"Loaded model from: {args.model_path}")
    print(f"Device: {device}\n")

    criterion = nn.CrossEntropyLoss()

    # Visualize top patches
    if not args.skip_patches:
        visualize_top_patches(
            model=model,
            df=df,
            feats_path=args.feats_path,
            wsi_dir=args.wsi_dir,
            num_classes=num_classes,
            class_labels=class_labels,
            criterion=criterion,
            device=device,
            model_type=args.model_type,
            output_dir=args.output_dir,
            top_k=args.top_k
        )

    # Visualize heatmaps
    if not args.skip_heatmaps:
        visualize_heatmaps(
            model=model,
            df=df,
            feats_path=args.feats_path,
            wsi_dir=args.wsi_dir,
            num_classes=num_classes,
            class_labels=class_labels,
            criterion=criterion,
            device=device,
            model_type=args.model_type,
            output_dir=args.output_dir
        )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
