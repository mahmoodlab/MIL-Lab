#!/usr/bin/env python3
"""
Evaluation utilities for MIL models.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    task_type: str = 'multiclass'
) -> Dict[str, Any]:
    """
    Evaluate a trained MIL model on test data.

    Args:
        model: Trained MIL model
        test_loader: Test data loader
        device: Device to evaluate on
        use_amp: Whether to use automatic mixed precision
        task_type: 'binary' or 'multiclass'

    Returns:
        Dictionary containing metrics and predictions.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for features, labels, *mask in tqdm(test_loader, desc='Evaluating'):
            features = features.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    results_dict, _ = model(
                        features,
                        loss_fn=criterion,
                        label=labels,
                    )
            else:
                results_dict, _ = model(
                    features,
                    loss_fn=criterion,
                    label=labels,
                )

            logits = results_dict['logits']
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())

    # Concatenate all logits and compute probabilities
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.softmax(all_logits, dim=1).numpy()

    metrics = calculate_metrics(
        all_labels, 
        all_preds, 
        y_prob=all_probs, 
        task_type=task_type
    )

    return {
        **metrics,
        'predictions': all_preds,
        'labels': all_labels,
    }


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[np.ndarray] = None,
    task_type: str = 'multiclass'
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, required for AUC)
        task_type: 'binary' or 'multiclass'

    Returns:
        Dictionary with metrics:
        - accuracy: Standard accuracy
        - balanced_accuracy: Balanced accuracy (macro recall)
        - f1_macro: F1 score (macro)
        - quadratic_kappa: Quadratic weighted Cohen's kappa
        - auc: Area Under the Curve (if y_prob provided)
        - confusion_matrix: Confusion matrix as numpy array
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'quadratic_kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            if task_type == 'binary':
                # For binary, use probability of positive class (index 1)
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_prob)
            else:
                # Multiclass: use One-vs-Rest macro average
                auc = roc_auc_score(
                    y_true, y_prob,
                    multi_class='ovr', average='macro'
                )
            metrics['auc'] = auc
        except ValueError:
            # Can fail if only one class is present in set
            metrics['auc'] = 0.0

    return metrics


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute standard MIL evaluation metrics (legacy wrapper).

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with basic metrics.
    """
    return calculate_metrics(y_true, y_pred)


def print_evaluation_results(results: Dict[str, Any], class_labels: Optional[List[str]] = None):
    """
    Print formatted evaluation results.

    Args:
        results: Results dictionary from evaluate()
        class_labels: Optional list of class label names
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nAccuracy:         {results['accuracy']:.4f}")
    print(f"Balanced Acc:     {results['balanced_accuracy']:.4f}")
    print(f"Quadratic Kappa:  {results['quadratic_kappa']:.4f}")

    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']

    if class_labels:
        # Print with labels
        max_label_len = max(len(str(l)) for l in class_labels)
        header = " " * (max_label_len + 2) + "  ".join(
            f"{l:>{max_label_len}}" for l in class_labels
        )
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{val:>{max_label_len}}" for val in row)
            print(f"{class_labels[i]:>{max_label_len}}  {row_str}")
    else:
        print(cm)

    print("=" * 70 + "\n")
