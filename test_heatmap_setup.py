"""
Test Heatmap Setup

Quick script to verify all dependencies and imports for heatmap generation.

Usage:
    python test_heatmap_setup.py
"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    errors = []

    # Core dependencies
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"✗ PyTorch: {e}")

    try:
        import h5py
        print(f"✓ h5py {h5py.__version__}")
    except ImportError as e:
        errors.append(f"✗ h5py: {e}")

    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"✗ pandas: {e}")

    try:
        import yaml
        print(f"✓ PyYAML")
    except ImportError as e:
        errors.append(f"✗ PyYAML: {e}")

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy: {e}")

    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError as e:
        errors.append(f"✗ Pillow: {e}")

    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"✗ OpenCV: {e}")

    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(f"✗ matplotlib: {e}")

    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError as e:
        errors.append(f"✗ SciPy: {e}")

    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError as e:
        errors.append(f"✗ tqdm: {e}")

    # OpenSlide (optional but recommended)
    try:
        import openslide
        print(f"✓ OpenSlide {openslide.__version__}")
    except ImportError as e:
        print(f"⚠ OpenSlide (optional): Not installed")
        print("  Install with: pip install openslide-python")

    return errors


def test_modules():
    """Test MIL-Lab modules"""
    print("\nTesting MIL-Lab modules...")
    errors = []

    # Visualization module
    try:
        from src.visualization import MILLabVisualizer, WholeSlideImage, draw_heatmap
        print("✓ src.visualization")
    except ImportError as e:
        errors.append(f"✗ src.visualization: {e}")

    # Builder
    try:
        from src.builder import create_model
        print("✓ src.builder")
    except ImportError as e:
        errors.append(f"✗ src.builder: {e}")

    # File utils
    try:
        from utils.file_utils import save_hdf5, load_hdf5
        print("✓ utils.file_utils")
    except ImportError as e:
        errors.append(f"✗ utils.file_utils: {e}")

    return errors


def test_files():
    """Test if required files exist"""
    print("\nTesting required files...")
    import os
    errors = []

    required_files = [
        'create_heatmaps.py',
        'create_process_list.py',
        'view_heatmap_sync.py',
        'src/visualization/heatmap_utils.py',
        'src/visualization/__init__.py',
        'utils/file_utils.py',
        'heatmaps/configs/config_template.yaml',
        'heatmaps/configs/config_abmil_panda.yaml',
        'HEATMAP_GENERATION.md',
        'HEATMAP_QUICKSTART.md',
    ]

    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath}")
        else:
            errors.append(f"✗ {filepath}: Not found")

    return errors


def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    import os
    errors = []

    required_dirs = [
        'heatmaps',
        'heatmaps/configs',
        'heatmaps/process_lists',
        'heatmaps/results',
        'src/visualization',
        'utils',
    ]

    for dirpath in required_dirs:
        if os.path.isdir(dirpath):
            print(f"✓ {dirpath}/")
        else:
            errors.append(f"✗ {dirpath}/: Not found")

    return errors


def main():
    print("="*60)
    print("HEATMAP GENERATION SETUP TEST")
    print("="*60)

    all_errors = []

    # Test imports
    all_errors.extend(test_imports())

    # Test modules
    all_errors.extend(test_modules())

    # Test files
    all_errors.extend(test_files())

    # Test directories
    all_errors.extend(test_directories())

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if all_errors:
        print(f"\n⚠ Found {len(all_errors)} issue(s):\n")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix the issues above before running heatmap generation.")
        return 1
    else:
        print("\n✓ All tests passed!")
        print("\nYou're ready to generate heatmaps!")
        print("\nNext steps:")
        print("  1. Create process list:")
        print("     python create_process_list.py --slide_dir /path/to/slides --output my_list.csv")
        print("\n  2. Edit config:")
        print("     cp heatmaps/configs/config_template.yaml heatmaps/configs/my_config.yaml")
        print("     # Edit paths in my_config.yaml")
        print("\n  3. Generate heatmaps:")
        print("     python create_heatmaps.py --config heatmaps/configs/my_config.yaml")
        print("\nSee HEATMAP_QUICKSTART.md for detailed instructions.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
