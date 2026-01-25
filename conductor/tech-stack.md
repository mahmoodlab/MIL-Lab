# Tech Stack

## Deep Learning & MIL
- **Framework:** PyTorch (>= 2.0.0)
- **Architectures:** HuggingFace Transformers, Timm
- **Multi-slide Fusion:** 
    - Early Fusion (Bag Concatenation via `GroupedMILDataset`)
    - Late Fusion (Embedding Averaging via `HierarchicalMILDataset` and `MIL` base class)

## Libraries & Frameworks
- **Model Architectures:** HuggingFace Transformers, Timm (for model backbones and vision utilities)
- **Computer Vision:** OpenCV, Scikit-image, OpenSlide (for WSI handling)
- **Data Analysis:** Pandas, NumPy, Scikit-learn
- **Scientific Computing:** SciPy

## MLOps & Monitoring
- **Experiment Tracking:** Weights & Biases (WandB)
- **Progress Tracking:** tqdm
- **Logging:** loguru

## Visualization
- **Plotting:** Matplotlib, Seaborn, Pillow
- **Interpretability:** Nystrom-attention (for specific attention mechanisms)

## Configuration & Environment
- **Configuration Management:** PyYAML, Omegaconf
- **Package Management:** Conda / Pip (setuptools)
