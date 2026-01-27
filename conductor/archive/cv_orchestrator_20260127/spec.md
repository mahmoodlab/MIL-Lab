# Specification: CV Execution Orchestrator and Aggregator

## Overview
This track aims to streamline the cross-validation workflow by creating a Python orchestrator. This script will automate the execution of multiple folds in parallel (on CPU) and aggregate the resulting metrics into a clear summary report.

## Functional Requirements
1.  **Parallel Execution:** Use Python `multiprocessing` to run cross-validation folds in parallel.
2.  **Configurable Concurrency:** Allow the user to specify the number of parallel workers (CPU threads).
3.  **Automatic Discovery:** Automatically find the `splits_N.json` files in a given split directory.
4.  **Integration:** Interface with the existing `run_mil_experiments_predefined_splits.py` script.
5.  **Aggregation:**
    -   Parse `test_results.json` from each fold's output directory.
    -   Calculate Mean and Standard Deviation for all numeric metrics.
6.  **Reporting:**
    -   Print a summary table to the console.
    -   Export aggregated results to a `cv_summary.csv` file.

## Technical Requirements
-   Create `scripts/run_cv_orchestrator.py`.
-   Use `subprocess.run` to call the training script.
-   Ensure clean error handling if a specific fold fails.

## Acceptance Criteria
-   A single command can trigger all CV folds for a given config and split directory.
-   Folds run in parallel according to the specified thread count.
-   A `cv_summary.csv` is generated with mean/std for metrics like Accuracy, Kappa, etc.
-   The console output displays a clear summary of the CV performance.

## Out of Scope
-   GPU parallelization (sticking to CPU threads as requested).
-   Plotting (e.g., boxplots of metrics across folds).
