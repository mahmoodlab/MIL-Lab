import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import MILTrainer
from training.config import TrainConfig, TaskType
from pathlib import Path

class MockModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    def forward(self, x, loss_fn=None, label=None):
        # x shape [batch, instances, features]
        # For simplicity, mean over instances
        if isinstance(x, list): # hierarchical
             x = torch.stack([f.mean(0) for f in x])
        else:
             x = x.mean(1)
        logits = self.fc(x)
        results = {'logits': logits}
        if loss_fn is not None and label is not None:
            results['loss'] = loss_fn(logits, label)
        return results, {}

def test_trainer_metric_selection(tmp_path):
    # Setup dummy data
    features = torch.randn(4, 5, 10) # 4 samples, 5 instances, 10 feats
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=2)
    
    config = TrainConfig(
        num_epochs=2,
        min_epochs=1,
        early_stopping_patience=1,
        early_stopping_metric="balanced_accuracy",
        task_type=TaskType.BINARY
    )
    
    model = MockModel(num_classes=2)
    trainer = MILTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        device=torch.device('cpu'),
        checkpoint_dir=str(tmp_path)
    )
    
    assert trainer._early_stopping_metric_name == "balanced_accuracy"
    
    history = trainer.fit()
    
    assert 'val_balanced_accuracy' in history
    assert 'val_accuracy' in history
    assert 'val_auc' in history
    assert 'val_f1_macro' in history

def test_trainer_auto_metric_selection():
    config_binary = TrainConfig(task_type=TaskType.BINARY, early_stopping_metric="auto")
    trainer_binary = MILTrainer(MockModel(), None, None, config_binary, torch.device('cpu'))
    assert trainer_binary._early_stopping_metric_name == "auc"
    
    config_multi = TrainConfig(task_type=TaskType.MULTICLASS, early_stopping_metric="auto")
    trainer_multi = MILTrainer(MockModel(), None, None, config_multi, torch.device('cpu'))
    assert trainer_multi._early_stopping_metric_name == "kappa"
