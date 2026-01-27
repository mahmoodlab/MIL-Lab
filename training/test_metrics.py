import pytest
import numpy as np
import torch
from training.evaluator import calculate_metrics

def test_calculate_metrics_binary():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    y_prob = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.4, 0.6],
        [0.3, 0.7]
    ])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob, task_type='binary')
    
    assert 'accuracy' in metrics
    assert 'balanced_accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'auc' in metrics
    assert 'quadratic_kappa' in metrics
    
    # Simple checks
    assert metrics['accuracy'] == 0.75
    # For binary, AUC should be roc_auc_score(y_true, y_prob[:, 1])
    # y_true = [0, 1, 0, 1], probs = [0.2, 0.9, 0.6, 0.7]
    # pairs: (0, 0.2), (1, 0.9), (0, 0.6), (1, 0.7)
    # 0.2 is less than 0.9 (correct), 0.2 less than 0.7 (correct)
    # 0.6 is less than 0.9 (correct), 0.6 less than 0.7 (correct)
    # AUC should be 1.0 here? No wait.
    # Sorted probs: 0.2 (y=0), 0.6 (y=0), 0.7 (y=1), 0.9 (y=1)
    # All 0s come before all 1s in prob order. AUC = 1.0.
    assert metrics['auc'] == 1.0

def test_calculate_metrics_multiclass():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 2, 2, 0]
    y_prob = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.3, 0.6],
        [0.1, 0.1, 0.8],
        [0.9, 0.05, 0.05]
    ])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob, task_type='multiclass')
    
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'auc' in metrics
    
    assert metrics['accuracy'] == 0.75
    assert metrics['auc'] > 0.5
