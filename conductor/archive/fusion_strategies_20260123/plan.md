# Implementation Plan - Multi-slide Fusion Strategies

## Phase 1: Configuration and Foundation [checkpoint: ac2e1cf]
- [x] Task: Update `DataConfig` in `training/config.py` to include `fusion` mode. ac2e1cf
- [x] Task: Update `MIL` base class in `src/models/mil_template.py` to handle hierarchical inputs in `forward`. ac2e1cf
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration and Foundation' (Protocol in workflow.md) ac2e1cf

## Phase 2: Trainer and Integration [checkpoint: 5c7ba38]
- [x] Task: Update `MILTrainer` in `training/trainer.py` to recursively move list-based features to device. 5c7ba38
- [x] Task: Update `train_mil.py` to select the correct dataset grouping based on `fusion` mode. 5c7ba38
- [x] Task: Conductor - User Manual Verification 'Phase 2: Trainer and Integration' (Protocol in workflow.md) 5c7ba38

## Phase 3: Testing and Verification [checkpoint: 422984]
- [x] Task: Create a test script to verify that Late Fusion produces expected output shapes and matches mock calculations. 422984
- [x] Task: Run a dummy training run with hierarchical data and late fusion. 422984
- [x] Task: Conductor - User Manual Verification 'Phase 3: Testing and Verification' (Protocol in workflow.md) 422984
