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

from .config import TrainConfig


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
            'val_accuracy': [],
            'val_kappa': [],
            'learning_rate': [],
        }
        self.best_val_kappa = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

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
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_kappa'].append(val_metrics['kappa'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Step scheduler
            self.scheduler.step()

            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            # Early stopping / checkpointing
            if val_metrics['kappa'] > self.best_val_kappa:
                self.best_val_kappa = val_metrics['kappa']
                self.best_epoch = epoch
                self.patience_counter = 0

                if self.checkpoint_dir:
                    self.save_checkpoint(self.checkpoint_dir / 'best_model.pth')
                    print(f"  >>> Saved best model (Val Kappa: {self.best_val_kappa:.4f})")
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

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Train]',
        )

        for features, labels, *mask in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Apply feature dropout
            features = self.feature_dropout(features)

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
        from sklearn.metrics import accuracy_score, cohen_kappa_score

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Val]',
        )

        with torch.no_grad():
            for features, labels, *mask in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)

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

        accuracy = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'kappa': kappa,
        }

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Print epoch summary."""
        print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"  Val Kappa:  {val_metrics['kappa']:.4f}")
        print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_kappa': self.best_val_kappa,
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
            self.best_val_kappa = checkpoint.get('best_val_kappa', -1.0)
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
