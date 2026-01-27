# Implementation Plan - Modular Metric Support (Refactor & Extend)

## Phase 1: Centralize & Extend Metric Calculation [checkpoint: 3269196]
- [x] Task: Update `training/evaluator.py` [b42682c]
    - [ ] Sub-task: Update `compute_metrics` (or create new `calculate_metrics`) to accept `y_prob` (probabilities) in addition to `y_true` and `y_pred`.
    - [ ] Sub-task: Implement **F1 Score** (Macro) calculation.
    - [ ] Sub-task: Implement **AUC** calculation (already in Trainer, needs moving to Evaluator) with handling for binary/multiclass.
    - [ ] Sub-task: Ensure `balanced_accuracy` and `kappa` are preserved.
    - [ ] Sub-task: Add unit tests for this new metric utility.
- [ ] Task: Conductor - User Manual Verification 'Centralize & Extend Metric Calculation' (Protocol in workflow.md)

## Phase 2: Configuration Update [checkpoint: f550f7e]
- [x] Task: Update `training/config.py` [cf49208]
    - [ ] Sub-task: Add `"f1"` and `"accuracy"` to the `early_stopping_metric` Literal type hint in `TrainConfig`.
    - [ ] Sub-task: Ensure `TaskType` is correctly utilized if needed for F1 (binary vs macro).
- [ ] Task: Conductor - User Manual Verification 'Configuration Update' (Protocol in workflow.md)

## Phase 3: Refactor Trainer
- [x] Task: Update `training/trainer.py` [fe19ecc]
    - [ ] Sub-task: Import `calculate_metrics` (or the updated `compute_metrics`) from `evaluator.py`.
    - [ ] Sub-task: Replace the inline metric calculation in `_validate_epoch` with the shared utility.
    - [ ] Sub-task: Update `_resolve_metric_name` to handle the new metrics (F1, Accuracy).
    - [ ] Sub-task: Ensure `_print_epoch_summary` logs all keys returned by the metric utility.
    - [ ] Sub-task: Verify `fit` loop correctly uses the `early_stopping_metric` for model saving.
- [ ] Task: Conductor - User Manual Verification 'Refactor Trainer' (Protocol in workflow.md)

## Phase 4: Verification
- [ ] Task: Create Verification Script `verify_trainer_metrics.py`
    - [ ] Sub-task: Create a script that instantiates `MILTrainer` with a simple dummy model and synthetic data.
    - [ ] Sub-task: Configure a run with `early_stopping_metric='balanced_accuracy'`.
    - [ ] Sub-task: Run the training and verify (via assertions) that the "best model" saved corresponds to the best balanced accuracy.
- [ ] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md)
