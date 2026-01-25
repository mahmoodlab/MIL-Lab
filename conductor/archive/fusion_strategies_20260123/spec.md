# Specification - Multi-slide Fusion Strategies

## Goal
Implement and standardize "Early Fusion" and "Late Fusion" strategies for multi-slide patient cases in MIL-Lab, following patterns seen in PathoBench.

## Definitions
1.  **Early Fusion (Pooled Patches):** All patches from all slides belonging to a patient are treated as a single "pseudo-slide". Aggregation (e.g., Attention) happens across the entire pool.
2.  **Late Fusion (Averaged Embeddings):** Each slide is processed independently through the MIL model's feature extractor to produce a slide-level embedding. These embeddings are then averaged (or otherwise aggregated) at the patient level before classification.

## Requirements
- Support both strategies via configuration.
- Leverage `GroupedMILDataset` for Early Fusion (most efficient).
- Enhance `MIL` base class and `MILTrainer` to support Late Fusion with `HierarchicalMILDataset`.
- Maintain compatibility with existing single-slide workflows.

## Proposed Implementation
- **Data Loading:**
    - `fusion=early` -> Use `MILDataset.concat_by('case_id')`.
    - `fusion=late` -> Use `MILDataset.group_by('case_id')`.
- **Model Logic (`MIL.forward`):**
    - If input is a list of tensors (Hierarchical), iterate through them, compute slide embeddings, and average.
- **Trainer:**
    - Recursively move list of tensors to device.
