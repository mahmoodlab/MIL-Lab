# Implementation Plan - CV Execution Orchestrator and Aggregator

## Phase 1: Orchestrator Script Development
- [x] Task: Create `scripts/run_cv_orchestrator.py`
    - [x] Sub-task: Define arguments: `--config`, `--split_dir`, `--output_dir`, `--workers`.
    - [x] Sub-task: Implement `find_folds(split_dir)` to identify available split files.
    - [x] Sub-task: Implement `run_fold_worker(fold_info)`: A worker function for `multiprocessing` that constructs and executes the `subprocess` call to `run_mil_experiments_predefined_splits.py`.
    - [x] Sub-task: Implement `aggregate_results(output_dir)`: Reads `test_results.json` from each fold, computes stats (mean/std) using Pandas, and saves `cv_summary.csv`.
- [ ] Task: Conductor - User Manual Verification 'Orchestrator Script Development' (Protocol in workflow.md)

## Phase 2: Testing & Integration [checkpoint: ecf9196]
- [x] Task: Create `tests/test_cv_orchestrator.py`
    - [x] Sub-task: Create a dummy test that mocks `subprocess.run` to verify the parallel logic and aggregation without actually training models.
    - [x] Sub-task: Create a dummy aggregation test with synthetic `test_results.json` files to verify CSV output correctness.
- [x] Task: Integration Verification
    - [x] Sub-task: Run the orchestrator on the dummy data/splits created in previous tracks (using `verify_cv_pipeline.py` setup logic if needed) with `--workers 2`.
- [x] Task: Conductor - User Manual Verification 'Testing & Integration' (Protocol in workflow.md) [ecf9196]
