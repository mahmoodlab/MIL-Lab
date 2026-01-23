# Initial Concept
The user is building a library for Multiple Instance Learning (MIL) models in computational pathology, specifically focusing on the FEATHER foundation model and standardized MIL implementations.

# Product Definition

## Target Users
- **Computational Pathology Researchers:** Seeking a standardized framework for MIL experimentation.
- **Clinical Data Scientists:** Integrating advanced pathology models into clinical workflows.
- **Pathologists:** Utilizing downstream analysis and inference tools powered by MIL.

## Goals
- **Model Standardization:** Standardize the initialization and training of various MIL models (e.g., ABMIL, CLAM, TransMIL).
- **Efficiency:** Provide "FEATHER" foundation models that are lightweight and efficient.
- **Benchmarking:** Enable comprehensive benchmarking across diverse datasets like TCGA.

## Key Features
- **Flexible Model Creation:** A `create_model()` interface supporting various patch encoders.
- **Encoder Integration:** Built-in support for patch foundation models such as UNI, CONCH, and GigaPath.
- **Benchmarking Suite:** Tools for evaluating models on morphological and molecular subtyping tasks.
- **Hierarchical Data Handling:** An interface designed to handle complex patient-to-slide relationships, including one slide per patient, multiple slides per patient, and multiple cores across multiple slides per patient.

## Success Metrics & Non-Functional Requirements
- **Reproducibility:** Ensuring benchmarking results are consistent and reproducible.
- **Extensibility:** Making it easy for users to add new MIL architectures and encoders.
- **Complex Data Support:** Robust handling of hierarchical pathology data structures (Patient -> Slide -> Core/Patch).
