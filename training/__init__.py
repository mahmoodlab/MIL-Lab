"""
Modular training infrastructure for MIL-Lab.

Usage:
    from training import ExperimentConfig, DataConfig, TrainConfig, MILTrainer, evaluate

    config = ExperimentConfig(
        data=DataConfig(labels_csv='...', features_dir='...'),
        train=TrainConfig(num_epochs=20),
        model_name='abmil.base.uni_v2.pc108-24k',
        num_classes=6,
    )

    trainer = MILTrainer(model, train_loader, val_loader, config.train, device)
    history = trainer.fit()
    results = evaluate(model, test_loader, device)
"""

from .config import (
    DataConfig,
    TrainConfig,
    ExperimentConfig,
)

from .trainer import MILTrainer

from .evaluator import (
    evaluate,
    compute_metrics,
    print_evaluation_results,
)


__all__ = [
    # Config
    'DataConfig',
    'TrainConfig',
    'ExperimentConfig',
    # Training
    'MILTrainer',
    # Evaluation
    'evaluate',
    'compute_metrics',
    'print_evaluation_results',
]
