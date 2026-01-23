# Specification - Hierarchical Data Loading Interface

## Goal
Implement a robust and flexible data loading interface that can handle complex patient-to-slide-to-core relationships commonly found in computational pathology.

## Requirements
- Support for "One patient, one slide" (standard MIL).
- Support for "One patient, multiple slides".
- Support for "One patient, multiple cores across multiple slides" (e.g., TMA or heterogeneous tissue).
- Seamless integration with existing `create_model()` and training workflows.
- Performance optimization for large-scale patch feature processing.

## Proposed Solution
- Extend or refactor `data_loading/dataset.py` to handle hierarchical indexing.
- Implement a mapping system that aggregates features at the patient level while preserving slide/core provenance.
- Ensure the PyTorch dataset adapter correctly batches these complex structures for MIL models.
