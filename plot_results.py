#!/usr/bin/env python3
"""
Standalone plotting script for MIL training results.

Usage:
    python plot_results.py --run-dir experiments/run_20240101_120000
    python plot_results.py --predictions path/to/predictions.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(labels, predictions, class_labels, title='Confusion Matrix', output_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_normalized_confusion_matrix(labels, predictions, class_labels, title='Normalized Confusion Matrix', output_path=None):
    """Plot and save normalized confusion matrix."""
    cm = confusion_matrix(labels, predictions, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot MIL training results')
    parser.add_argument('--run-dir', type=str, help='Path to run directory containing predictions.npz')
    parser.add_argument('--predictions', type=str, help='Direct path to predictions.npz file')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()

    # Find predictions file
    if args.predictions:
        pred_path = Path(args.predictions)
    elif args.run_dir:
        pred_path = Path(args.run_dir) / 'predictions.npz'
    else:
        print("Error: Provide --run-dir or --predictions")
        return

    if not pred_path.exists():
        print(f"Error: {pred_path} not found")
        return

    # Load predictions
    data = np.load(pred_path, allow_pickle=True)
    labels = data['labels']
    predictions = data['predictions']
    class_labels = list(data['class_labels'])

    print(f"Loaded {len(labels)} predictions")
    print(f"Classes: {class_labels}")

    # Output directory
    out_dir = pred_path.parent

    # Plot confusion matrix
    plot_confusion_matrix(
        labels, predictions, class_labels,
        title='Confusion Matrix',
        output_path=None if args.show else out_dir / 'confusion_matrix.png'
    )

    # Plot normalized confusion matrix
    plot_normalized_confusion_matrix(
        labels, predictions, class_labels,
        title='Normalized Confusion Matrix',
        output_path=None if args.show else out_dir / 'confusion_matrix_normalized.png'
    )

    print("Done!")


if __name__ == '__main__':
    main()
