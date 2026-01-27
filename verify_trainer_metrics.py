import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import MILTrainer
from training.config import TrainConfig, TaskType
from pathlib import Path
import shutil

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
    def forward(self, x, loss_fn=None, label=None):
        x = x.mean(1)
        logits = self.fc(x)
        results = {'logits': logits}
        if loss_fn is not None and label is not None:
            results['loss'] = loss_fn(logits, label)
        return results, {}

def verify():
    print("Starting trainer metric verification...")
    checkpoint_dir = Path("test_checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()

    # Synthetic data
    features = torch.randn(10, 3, 5) # 10 samples
    labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=2)

    # Config for balanced_accuracy
    config = TrainConfig(
        num_epochs=3,
        min_epochs=1,
        early_stopping_patience=10,
        early_stopping_metric="balanced_accuracy",
        task_type=TaskType.BINARY
    )

    model = SimpleModel()
    trainer = MILTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        device=torch.device('cpu'),
        checkpoint_dir=str(checkpoint_dir)
    )

    print(f"Training with early_stopping_metric: {config.early_stopping_metric}")
    history = trainer.fit()

    # Check best model saving
    best_model_path = checkpoint_dir / "best_model.pth"
    assert best_model_path.exists(), "Best model checkpoint was not saved"
    
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
    assert 'early_stopping_metric_name' in checkpoint
    assert checkpoint['early_stopping_metric_name'] == "balanced_accuracy"
    
    print("Verification successful: Metrics tracked and checkpoint saved correctly.")
    
    # Cleanup
    shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    verify()
