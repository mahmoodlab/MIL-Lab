#!/usr/bin/env python3
"""
Configuration dataclasses for MIL training.

Simple Python dataclasses that can be easily extended to support
YAML/JSON loading or MLflow parameter logging.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading."""
    labels_csv: str
    features_dir: str
    split_column: Optional[str] = None  # If set, use this column for splits
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42
    num_workers: int = 4
    hierarchical: bool = False
    group_column: str = 'case_id'

    def __post_init__(self):
        # Validate paths exist
        if not Path(self.labels_csv).exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")
        if not Path(self.features_dir).exists():
            raise FileNotFoundError(f"Features dir not found: {self.features_dir}")


@dataclass
class TrainConfig:
    """Configuration for training loop."""
    num_epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    feature_dropout: float = 0.1
    model_dropout: float = 0.25
    early_stopping_patience: int = 100  # Match original script
    min_epochs: int = 10  # Match original script
    max_grad_norm: float = 1.0
    use_amp: bool = True
    weighted_sampling: bool = True
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    data: DataConfig
    train: TrainConfig
    model_name: str
    num_classes: int
    output_dir: str = 'experiments'
    experiment_name: Optional[str] = None

    def __post_init__(self):
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Generate experiment name if not provided
        if self.experiment_name is None:
            model_short = self.model_name.replace('.', '_')
            self.experiment_name = f"{model_short}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary (useful for MLflow logging)."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    result[f"{key}.{sub_key}"] = sub_value
            else:
                result[key] = value
        return result

    def save(self, path: str):
        """Save config to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            data=DataConfig(**data['data']),
            train=TrainConfig(**data['train']),
            model_name=data['model_name'],
            num_classes=data['num_classes'],
            output_dir=data.get('output_dir', 'experiments'),
            experiment_name=data.get('experiment_name'),
        )
