#!/usr/bin/env python3
"""
Standalone plotting script for cross-validation results.

Reads per-fold prediction .npz files saved by run_mil_experiments_cv.py
and produces:
  - Multi-fold ROC curves (binary classification)
  - Confusion matrix grid (one per fold)
  - Misclassified slide report (console + CSV)

Usage:
    python plot_cv_results.py --results-dir experiment_results_cv/run_XXXXXXXX_XXXXXX
    python plot_cv_results.py --results-dir experiment_results_cv/run_XXXXXXXX_XXXXXX --show
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def load_fold_data(results_dir):
    """
    Load all per-fold .npz prediction files from a results directory.

    Args:
        results_dir: Path to directory containing fold_*_predictions.npz

    Returns:
        List of dicts, each with keys: y_true, y_pred, y_prob, slide_ids, class_labels, fold_num
    """
    results_dir = Path(results_dir)
    npz_files = sorted(results_dir.glob('fold_*_predictions.npz'),
                       key=lambda p: int(p.stem.split('_')[1]))

    if not npz_files:
        raise FileNotFoundError(f"No fold_*_predictions.npz files found in {results_dir}")

    fold_data_list = []
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        fold_num = int(npz_path.stem.split('_')[1])
        fold_data_list.append({
            'y_true': data['y_true'],
            'y_pred': data['y_pred'],
            'y_prob': data['y_prob'],
            'slide_ids': data['slide_ids'],
            'class_labels': list(data['class_labels']),
            'fold_num': fold_num,
        })

    print(f"Loaded {len(fold_data_list)} folds from {results_dir}")
    return fold_data_list


def plot_roc_curves_binary(fold_data_list, output_path=None):
    """
    Plot binary ROC curves for all folds on a single figure.

    One thin curve per fold, plus a thick mean curve with ±1 std shaded band.

    Args:
        fold_data_list: List of fold data dicts (from load_fold_data)
        output_path: Path to save figure. If None, plt.show() is called.
    """
    mean_fpr = np.linspace(0, 1, 100)
    all_tprs = []
    all_aucs = []

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    for fold_data in fold_data_list:
        y_true = fold_data['y_true']
        y_prob = fold_data['y_prob']
        fold_num = fold_data['fold_num']

        # For binary: use probability of positive class (column 1)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_score = y_prob[:, 1]
        elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
            print(f"  Warning: Fold {fold_num} has {y_prob.shape[1]} classes, "
                  "expected binary. Using class 1 probability.")
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        all_aucs.append(auc_val)

        # Interpolate TPR onto shared FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        all_tprs.append(interp_tpr)

        ax.plot(fpr, tpr, alpha=0.4, linewidth=1,
                label=f'Fold {fold_num} (AUC = {auc_val:.3f})')

    # Mean ROC curve
    mean_tpr = np.mean(all_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)

    ax.plot(mean_fpr, mean_tpr, color='b', linewidth=2.5,
            label=f'Mean (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})')

    # ±1 std band
    std_tpr = np.std(all_tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.15,
                    label='$\\pm$ 1 std')

    # Diagonal
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves Across Folds', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrices(fold_data_list, output_path=None):
    """
    Plot a grid of confusion matrices, one per fold.

    Args:
        fold_data_list: List of fold data dicts
        output_path: Path to save figure. If None, plt.show() is called.
    """
    n_folds = len(fold_data_list)
    class_labels = fold_data_list[0]['class_labels']

    # Layout: up to 3 per row
    ncols = min(n_folds, 3)
    nrows = (n_folds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_folds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, fold_data in enumerate(fold_data_list):
        cm = confusion_matrix(fold_data['y_true'], fold_data['y_pred'])
        plot_labels = [l.replace(' ', '\n') for l in class_labels]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=plot_labels, yticklabels=plot_labels,
                    ax=axes[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        axes[i].set_title(f'Fold {fold_data["fold_num"]}')

    # Hide unused subplots
    for j in range(n_folds, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Confusion Matrices Per Fold', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def report_misclassified(fold_data_list, output_path=None):
    """
    Print misclassified slides per fold and save to CSV.

    Args:
        fold_data_list: List of fold data dicts
        output_path: Path to save CSV. If None, only prints to console.
    """
    rows = []

    for fold_data in fold_data_list:
        fold_num = fold_data['fold_num']
        y_true = fold_data['y_true']
        y_pred = fold_data['y_pred']
        slide_ids = fold_data['slide_ids']
        class_labels = fold_data['class_labels']

        mask = y_true != y_pred
        n_misclassified = mask.sum()

        if n_misclassified == 0:
            print(f"Fold {fold_num}: No misclassifications")
            continue

        print(f"\nFold {fold_num}: {n_misclassified} misclassified")
        for sid, tl, pl in zip(slide_ids[mask], y_true[mask], y_pred[mask]):
            true_name = class_labels[tl]
            pred_name = class_labels[pl]
            print(f"  {sid} | True: {true_name} | Predicted: {pred_name}")
            rows.append({
                'fold': fold_num,
                'slide_id': sid,
                'true_label': int(tl),
                'predicted_label': int(pl),
                'true_class': true_name,
                'predicted_class': pred_name,
            })

    if output_path and rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\nMisclassified report saved to: {output_path}")
    elif not rows:
        print("\nNo misclassifications across any fold.")


def main():
    parser = argparse.ArgumentParser(description='Plot cross-validation results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to CV results directory containing fold_*_predictions.npz')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively instead of saving')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return

    # Load all fold data
    fold_data_list = load_fold_data(results_dir)

    save_path = None if args.show else results_dir

    # ROC curves (binary)
    print("\n" + "=" * 60)
    print("ROC CURVES")
    print("=" * 60)
    plot_roc_curves_binary(
        fold_data_list,
        output_path=save_path / 'roc_curves.png' if save_path else None,
    )

    # Confusion matrices
    print("\n" + "=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)
    plot_confusion_matrices(
        fold_data_list,
        output_path=save_path / 'confusion_matrices.png' if save_path else None,
    )

    # Misclassified report
    print("\n" + "=" * 60)
    print("MISCLASSIFIED SLIDES")
    print("=" * 60)
    report_misclassified(
        fold_data_list,
        output_path=save_path / 'misclassified.csv' if save_path else None,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
