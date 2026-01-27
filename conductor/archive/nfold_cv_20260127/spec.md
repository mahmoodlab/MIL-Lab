# Specification: Patient-Level N-Fold Cross-Validation Splits

## Overview
This track implements a robust K-fold cross-validation infrastructure for MIL training. To ensure scientific rigor and prevent data leakage, the splitting will be performed at the patient (case) level, ensuring all slides from a single patient remain within the same fold.

## Functional Requirements
1.  **Patient-Level Splitting:** Splits must be performed on unique Patient/Case IDs to prevent leakage of multi-slide cases.
2.  **K-Fold Generation:** Support for generating `N` folds (default 5 or 10).
3.  **Internal Validation Set:** For each fold, the training portion must be further split into `train` and `val` (for early stopping), while the fold's test portion remains a true `test` set.
4.  **Configurable Ratios:** The internal Train/Val split ratio should be configurable.
5.  **Output Format:** Splits must be saved as **JSON** files.
6.  **Storage:** All split files will be stored in a dedicated `splits/` directory.
7.  **Dataset Integration:** Update or create a data-loading utility that can load a specific fold's JSON file to prepare the `MILDataset`.

## Functional Requirements (Technical)
-   Create a script `scripts/create_splits.py` to handle the generation logic.
-   The JSON format should clearly map `case_id` to its respective set (`train`, `val`, `test`) for each fold.

## Non-Functional Requirements
-   **Reproducibility:** Splits must be deterministic given a random seed.
-   **Traceability:** Include metadata in the JSON (seed, source CSV, date).

## Acceptance Criteria
-   A command-line script exists to generate `N` JSON split files.
-   JSON files correctly partition Patient IDs without overlap between test and train/val.
-   The `MILDataset` or a wrapper can successfully filter data based on these JSON files.
-   Slides belonging to the same `case_id` always appear in the same split.

## Out of Scope
-   Automating the execution of all `N` folds in a single command (this can be done via shell scripts later).
-   Modifying the model architecture.
