# Implementation Plan - Hierarchical Data Loading

## Phase 1: Analysis and Design [checkpoint: aeadfed]
- [x] Task: Review existing `data_loading/dataset.py` and `data_loading/feature_loader.py` for current slide/patient assumptions. aeadfed
- [x] Task: Define the data schema (CSV or JSON) that captures patient-slide-core hierarchies. aeadfed
- [x] Task: Conductor - User Manual Verification 'Phase 1: Analysis and Design' (Protocol in workflow.md) aeadfed

## Phase 2: Core Implementation [checkpoint: ac2e1cf]
- [x] Task: Implement a Hierarchical Dataset class that supports patient-level aggregation. ac2e1cf
    - [x] Write unit tests for the hierarchical indexer. ac2e1cf
    - [x] Implement the logic to group features by patient across multiple slides/cores. ac2e1cf
- [x] Task: Update the PyTorch adapter to handle variable-length patches from multiple sources per patient. ac2e1cf
    - [x] Write tests for the collation function. ac2e1cf
    - [x] Implement robust batching logic. ac2e1cf
- [x] Task: Conductor - User Manual Verification 'Phase 2: Core Implementation' (Protocol in workflow.md) ac2e1cf

## Phase 3: Integration and Benchmarking
- [ ] Task: Integrate the new loader with `train_mil.py` or a dedicated experiment runner.
- [ ] Task: Verify that attention heatmaps can still be mapped back to individual slides/cores.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Benchmarking' (Protocol in workflow.md)
