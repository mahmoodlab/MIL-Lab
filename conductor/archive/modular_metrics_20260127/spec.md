# Specification: Modular Metric Support in MILTrainer

## Overview
Currently, `MILTrainer` is hardcoded to use Quadratic Kappa for early stopping and model selection. This track aims to refactor the trainer to support multiple evaluation metrics (Balanced Accuracy, AUC, F1, Kappa, Accuracy) and allow the user to specify which metric to use via the `TrainConfig`.

## Functional Requirements
1.  **Configurable Metric Selection:** Add an `early_stopping_metric` field to `TrainConfig`.
2.  **Support for Multiple Metrics:** The trainer must calculate and track:
    -   Standard Accuracy
    -   Balanced Accuracy
    -   Quadratic Kappa
    -   F1 Score (Macro)
    -   AUC (Area Under the Curve)
3.  **Modular Evaluation:** Refactor `_validate_epoch` to utilize a centralized metric computation utility (ideally leveraging or extending `evaluator.py`).
4.  **Flexible Early Stopping:** Update the early stopping logic to monitor the user-specified metric from `TrainConfig`.
5.  **History Tracking:** The `history` dictionary should automatically include all calculated metrics.

## Non-Functional Requirements
-   **Backward Compatibility:** Default the `early_stopping_metric` to `quadratic_kappa` to maintain current behavior.
-   **Efficiency:** Ensure AUC calculation does not significantly slow down validation (requires collecting logits/probabilities, not just hard predictions).

## Acceptance Criteria
-   The trainer successfully completes a run using `balanced_accuracy` and other metrics for early stopping.
-   All requested metrics are visible in the printed epoch summary and the `history` object.
-   The "Best Model" is saved based on the configured metric.
-   The code follows existing patterns in `trainer.py` and `evaluator.py`.

## Out of Scope
-   Changing the training loss function (staying with CrossEntropy for now).
-   Implementing multi-class specific AUC strategies (e.g., One-vs-Rest) unless trivial to do so via `sklearn`.
