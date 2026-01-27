# Implementation Plan - Patient-Level N-Fold Cross-Validation Splits

## Phase 1: Split Generation Infrastructure
- [ ] Task: Create Split Generation Script `scripts/create_splits.py`
    - [ ] Sub-task: Implement logic to read the main labels CSV and group by `case_id`.
    - [ ] Sub-task: Use `sklearn.model_selection.KFold` or `StratifiedKFold` to generate K partitions of unique `case_id`s.
    - [ ] Sub-task: For each fold, further split the "training" portion into `train` and `val` sets based on a configurable ratio.
    - [ ] Sub-task: Implement a JSON export function that saves the mapping of `case_id` to its set (`train`, `val`, `test`).
    - [ ] Sub-task: Ensure the script accepts command-line arguments for `n_folds`, `val_ratio`, `seed`, and `label_csv`.
- [ ] Task: Write Tests for Split Logic `tests/test_splitting.py`
    - [ ] Sub-task: Verify that no `case_id` overlaps between sets in a single fold.
    - [ ] Sub-task: Verify that all slides for a given `case_id` are assigned to the same set.
    - [ ] Sub-task: Verify reproducibility with a fixed seed.
- [ ] Task: Conductor - User Manual Verification 'Split Generation Infrastructure' (Protocol in workflow.md)

## Phase 2: Data Loading Integration
- [ ] Task: Update `data_loading/dataset.py` (or adapter) to support Split JSONs
    - [ ] Sub-task: Add a method or parameter to filter the dataframe based on a provided split JSON and the desired set name.
    - [ ] Sub-task: Ensure this works correctly with the existing `MILDataset` and `HierarchicalMILDataset`.
- [ ] Task: Write Tests for Dataset Filtering `tests/test_dataset_splits.py`
    - [ ] Sub-task: Load a generated split JSON and verify that the `DataLoader` only yields samples from the correct set.
- [ ] Task: Conductor - User Manual Verification 'Data Loading Integration' (Protocol in workflow.md)

## Phase 3: Training Script Update
- [ ] Task: Update `run_mil.py` (or create a CV-aware version) to accept split files
    - [ ] Sub-task: Add a `--split_dir` and `--fold` argument to the script.
    - [ ] Sub-task: Implement logic to load the corresponding `split_<fold>.json` and pass it to the data loader.
- [ ] Task: Integration Verification
    - [ ] Sub-task: Run a dummy training run using one of the generated folds to ensure the pipeline is end-to-end functional.
- [ ] Task: Conductor - User Manual Verification 'Training Script Update' (Protocol in workflow.md)
