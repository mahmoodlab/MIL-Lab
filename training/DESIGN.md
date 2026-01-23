# Training Module - Design Specification

## Overview

The `training` module provides a modular training infrastructure for MIL models with support for early stopping, mixed precision training, and MLflow-ready logging.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ExperimentConfig                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  DataConfig  │  │ TrainConfig  │  │  model_name  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      MILTrainer                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • AMP (mixed precision)                           │  │
│  │ • Gradient clipping                               │  │
│  │ • Early stopping (validation kappa)               │  │
│  │ • Best model checkpointing                        │  │
│  │ • History tracking                                │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      Evaluator                           │
│  • evaluate() - Full model evaluation                    │
│  • compute_metrics() - accuracy, balanced_acc, kappa     │
│  • print_evaluation_results() - Formatted output         │
└─────────────────────────────────────────────────────────┘
```

## Components

### Configuration Classes

#### `DataConfig`
```python
@dataclass
class DataConfig:
    labels_csv: str
    features_dir: str
    split_column: Optional[str] = None
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42
    num_workers: int = 4
```

#### `TrainConfig`
```python
@dataclass
class TrainConfig:
    num_epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    feature_dropout: float = 0.1
    model_dropout: float = 0.25
    early_stopping_patience: int = 100
    min_epochs: int = 10
    max_grad_norm: float = 1.0
    use_amp: bool = True
    weighted_sampling: bool = True
    seed: int = 42
```

#### `ExperimentConfig`
- Combines `DataConfig` + `TrainConfig` + model settings
- JSON save/load for reproducibility
- `to_dict()` for MLflow parameter logging

### MILTrainer

**Features**:
- Automatic Mixed Precision (AMP) training
- Gradient clipping (`max_grad_norm`)
- Feature dropout (applied to input features)
- Early stopping based on validation quadratic kappa
- Best model checkpointing
- History tracking (train_loss, val_loss, val_accuracy, val_kappa, lr)

**Methods**:
- `fit()` → Train with early stopping, returns history
- `save_checkpoint(path)` → Save full training state
- `load_checkpoint(path, weights_only)` → Restore state
- `load_best_model()` → Load best checkpoint for evaluation

### Evaluator

**Functions**:
- `evaluate(model, test_loader, device)` → Returns metrics dict
- `compute_metrics(y_true, y_pred)` → accuracy, balanced_acc, kappa, confusion_matrix
- `print_evaluation_results(results, class_labels)` → Formatted output

## Design Decisions

### 1. Dataclass Configuration

**Chosen**: Python dataclasses over YAML/JSON configs

**Rationale**:
- Type hints and IDE support
- Easy to extend
- Can serialize to JSON when needed
- No external dependencies

### 2. Early Stopping Metric

**Chosen**: Quadratic weighted Cohen's kappa

**Rationale**:
- Standard metric for ordinal classification (ISUP grades)
- Accounts for class imbalance
- Penalizes predictions far from true label more heavily

### 3. Checkpoint Contents

**Saved**:
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `best_val_kappa`
- `best_epoch`
- `history`

**Not saved** (by design):
- Current epoch (resume not fully supported yet)
- Patience counter
- AMP scaler state

### 4. Mixed Precision

**Default**: Enabled (`use_amp=True`)

**Rationale**:
- Significant speedup on modern GPUs
- Minimal accuracy impact for MIL
- Can be disabled for debugging

## Usage

### Basic Training
```python
from training import TrainConfig, MILTrainer, evaluate

config = TrainConfig(num_epochs=20)
trainer = MILTrainer(model, train_loader, val_loader, config, device, checkpoint_dir='runs/')
history = trainer.fit()

trainer.load_best_model()
results = evaluate(model, test_loader, device)
```

### With ExperimentConfig
```python
from training import ExperimentConfig, DataConfig, TrainConfig

config = ExperimentConfig(
    data=DataConfig(labels_csv='labels.csv', features_dir='features/'),
    train=TrainConfig(num_epochs=20),
    model_name='abmil.base.uni_v2.pc108-24k',
    num_classes=6,
)

config.save('experiment_config.json')
loaded = ExperimentConfig.load('experiment_config.json')
```

## Output Structure

```
experiments/run_YYYYMMDD_HHMMSS/
├── config.json           # ExperimentConfig
├── best_model.pth        # Best checkpoint
├── results.json          # Final metrics
└── confusion_matrix.png  # Visualization
```

## Future Extensions

1. **MLflow Integration**: Log metrics, params, artifacts
2. **Resume Training**: Full checkpoint restore with epoch counter
3. **Learning Rate Finder**: Auto-find optimal LR
4. **Multi-GPU**: DataParallel/DistributedDataParallel support
5. **Callbacks**: Custom callbacks for logging, early stopping, etc.
