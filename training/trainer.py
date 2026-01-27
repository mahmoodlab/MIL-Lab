#!/usr/bin/env python3
"""
MIL Trainer with early stopping, AMP, and MLflow-ready history tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config import TrainConfig, TaskType
from .evaluator import calculate_metrics


class MILTrainer:
    """
    Trainer for MIL models with built-in early stopping.

    Features:
    - Automatic mixed precision (AMP) training
    - Gradient clipping
    - Early stopping based on validation metric
    - Best model checkpointing
    - History tracking (MLflow-ready)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Args:
            model: MIL model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Feature dropout (applied to input features)
        self.feature_dropout = nn.Dropout(p=config.feature_dropout).to(device)

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6,
        )
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        self.best_val_metric = -1.0
        self.best_epoch = 0
        self.patience_counter = 0
        self._early_stopping_metric_name = self._resolve_metric_name()

    def _resolve_metric_name(self) -> str:
        """Resolve the metric name based on config."""
        metric = self.config.early_stopping_metric
        if metric == "auto":
            # Auto-select: AUC for binary, kappa for multiclass
            if self.config.task_type == TaskType.BINARY:
                return "auc"
            else:
                return "kappa"
        return metric

    def _get_early_stopping_metric(self, val_metrics: Dict[str, float]) -> float:
        """Get the metric value to use for early stopping."""
        # Use a mapping that aligns with calculate_metrics keys
        metric_key = self._early_stopping_metric_name
        if metric_key == "kappa":
            metric_key = "quadratic_kappa"
        
        return val_metrics.get(metric_key, val_metrics.get('balanced_accuracy', 0.0))

    def fit(self) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.

        Returns:
            History dictionary with metrics for each epoch.
        """
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self._train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])

            # Validation
            val_metrics = self._validate_epoch(epoch)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Record all other metrics (accuracy, auc, etc.) in history
            for key, val in val_metrics.items():
                if key != 'loss' and key != 'confusion_matrix':
                    history_key = f'val_{key}'
                    if history_key not in self.history:
                        self.history[history_key] = []
                    self.history[history_key].append(val)

            # Step scheduler
            self.scheduler.step()

            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            # Early stopping / checkpointing
            current_metric = self._get_early_stopping_metric(val_metrics)
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0

                if self.checkpoint_dir:
                    self.save_checkpoint(self.checkpoint_dir / 'best_model.pth')
                    metric_display = self._early_stopping_metric_name.upper()
                    print(f"  >>> Saved best model (Val {metric_display}: {self.best_val_metric:.4f})")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config.early_stopping_patience}")

            # Check early stopping
            if (self.patience_counter >= self.config.early_stopping_patience
                    and (epoch + 1) >= self.config.min_epochs):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

            print("-" * 70)

        return self.history

    @staticmethod
    def to_device(data: Any, device: torch.device) -> Any:
        """Recursively move data to device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, (list, tuple)):
            return [MILTrainer.to_device(x, device) for x in data]
        elif isinstance(data, dict):
            return {k: MILTrainer.to_device(v, device) for k, v in data.items()}
        return data

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Train]',
        )

        for features, labels, *mask in pbar:
            features = self.to_device(features, self.device)
            labels = self.to_device(labels, self.device)

            # Apply feature dropout (only if features is a tensor)
            if isinstance(features, torch.Tensor):
                features = self.feature_dropout(features)
            else:
                # For hierarchical, apply dropout to each slide tensor
                features = [self.feature_dropout(f) for f in features]

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.config.use_amp:
                with torch.amp.autocast('cuda'):
                    results_dict, _ = self.model(
                        features,
                        loss_fn=self.criterion,
                        label=labels,
                    )
                    loss = results_dict['loss']

                # Scaled backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results_dict, _ = self.model(
                    features,
                    loss_fn=self.criterion,
                    label=labels,
                )
                loss = results_dict['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm,
                )
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return {'loss': total_loss / len(self.train_loader)}

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_logits = []

        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Val]',
        )

        with torch.no_grad():
            for features, labels, *mask in pbar:
                features = self.to_device(features, self.device)
                labels = self.to_device(labels, self.device)

                if self.config.use_amp:
                    with torch.amp.autocast('cuda'):
                        results_dict, _ = self.model(
                            features,
                            loss_fn=self.criterion,
                            label=labels,
                        )
                else:
                    results_dict, _ = self.model(
                        features,
                        loss_fn=self.criterion,
                        label=labels,
                    )

                loss = results_dict['loss']
                logits = results_dict['logits']

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.append(logits.cpu())

        # Concatenate logits for probability computation
        all_logits = torch.cat(all_logits, dim=0)
        all_probs = torch.softmax(all_logits, dim=1).numpy()

        # Calculate metrics using shared utility
        metrics = calculate_metrics(
            all_labels,
            all_preds,
            y_prob=all_probs,
            task_type=self.config.task_type.value
        )
        
        # Add loss
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Print epoch summary."""
        print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
        print(f"  Train Loss:    {train_metrics['loss']:.4f}")
        print(f"  Val Loss:      {val_metrics['loss']:.4f}")
        
        # Print all other metrics
        for key, val in val_metrics.items():
            if key not in ['loss', 'confusion_matrix']:
                # Format key for display
                display_key = key.replace('_', ' ').title()
                print(f"  Val {display_key:<10}: {val:.4f}")
                
        print(f"  LR:            {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Early Stop Metric ({self._early_stopping_metric_name}): "
              f"{self._get_early_stopping_metric(val_metrics):.4f}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'early_stopping_metric_name': self._early_stopping_metric_name,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str, weights_only: bool = True):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            weights_only: If True, only load model weights (still loads full checkpoint file)
        """
        # weights_only=False for torch.load because our checkpoint contains numpy arrays in history
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if not weights_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_metric = checkpoint.get('best_val_metric', -1.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.history = checkpoint.get('history', self.history)

    def load_best_model(self):
        """Load the best model from checkpoint."""
        if self.checkpoint_dir:
            best_path = self.checkpoint_dir / 'best_model.pth'
            if best_path.exists():
                self.load_checkpoint(best_path, weights_only=True)
                print(f"Loaded best model from epoch {self.best_epoch + 1}")
            else:
                print("Warning: No best model checkpoint found.")
