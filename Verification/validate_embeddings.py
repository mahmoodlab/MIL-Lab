#!/usr/bin/env python3
"""
Embedding Validation Script
Validates h5 embedding files and compares old vs new directories
"""

import h5py
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import torch

# Configuration
OLD_FEATS_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/panda/'
NEW_FEATS_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/'

def validate_h5_file(h5_path, verbose=False):
    """
    Validate a single h5 file and return validation results

    Returns:
        dict with validation results
    """
    result = {
        'path': h5_path,
        'valid': True,
        'errors': [],
        'warnings': [],
        'features_shape': None,
        'coords_shape': None,
        'has_features': False,
        'has_coords': False,
        'dtype': None,
    }

    try:
        with h5py.File(h5_path, 'r') as f:
            # Check required keys
            if 'features' not in f:
                result['valid'] = False
                result['errors'].append("Missing 'features' key")
            else:
                result['has_features'] = True
                result['features_shape'] = f['features'].shape
                result['dtype'] = str(f['features'].dtype)

                # Validate features shape
                features_shape = f['features'].shape
                if len(features_shape) != 3:
                    result['valid'] = False
                    result['errors'].append(f"Expected 3D features, got {len(features_shape)}D: {features_shape}")
                elif features_shape[0] != 1:
                    result['warnings'].append(f"First dimension is {features_shape[0]}, expected 1")
                elif features_shape[2] != 1536:
                    result['valid'] = False
                    result['errors'].append(f"Expected 1536 feature dimensions, got {features_shape[2]}")

                # Test if data can be loaded
                try:
                    test_data = f['features'][:]
                    result['can_load'] = True

                    # Test torch conversion
                    try:
                        tensor = torch.from_numpy(test_data)
                        result['can_convert_torch'] = True

                        # Test squeeze and clone (what the dataset does)
                        squeezed = tensor.squeeze(0)
                        cloned = squeezed.clone()
                        result['can_process'] = True
                        result['final_shape'] = tuple(cloned.shape)

                    except Exception as e:
                        result['valid'] = False
                        result['errors'].append(f"Torch conversion failed: {e}")
                        result['can_convert_torch'] = False

                except Exception as e:
                    result['valid'] = False
                    result['errors'].append(f"Cannot load features data: {e}")
                    result['can_load'] = False

            # Check coords
            if 'coords_patching' in f:
                result['has_coords'] = True
                result['coords_shape'] = f['coords_patching'].shape

                # Validate coords shape matches features
                if result['has_features']:
                    features_patches = f['features'].shape[1]
                    coords_patches = f['coords_patching'].shape[0]
                    if features_patches != coords_patches:
                        result['warnings'].append(
                            f"Coords/features mismatch: {coords_patches} coords vs {features_patches} features"
                        )
            else:
                result['warnings'].append("Missing 'coords_patching' key")

            # Check other keys
            result['keys'] = list(f.keys())

    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to open file: {e}")

    if verbose and (not result['valid'] or result['warnings']):
        print(f"\n{os.path.basename(h5_path)}:")
        if result['errors']:
            print(f"  ❌ ERRORS: {result['errors']}")
        if result['warnings']:
            print(f"  ⚠️  WARNINGS: {result['warnings']}")

    return result


def validate_directory(feats_path, name="Directory", max_files=None, verbose=False):
    """Validate all h5 files in a directory"""

    print(f"\n{'='*70}")
    print(f"VALIDATING {name.upper()}")
    print(f"{'='*70}")
    print(f"Path: {feats_path}")

    # Find all h5 files
    h5_files = glob(os.path.join(feats_path, '*.h5'))

    if not h5_files:
        print(f"❌ No h5 files found in {feats_path}")
        return None

    print(f"Found {len(h5_files)} h5 files")

    # Limit files if specified
    if max_files:
        h5_files = h5_files[:max_files]
        print(f"Validating first {max_files} files...")
    else:
        print(f"Validating all files...")

    # Validate all files
    results = []
    for h5_file in tqdm(h5_files, desc=f"Validating {name}"):
        result = validate_h5_file(h5_file, verbose=verbose)
        results.append(result)

    # Summary statistics
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    warning_count = sum(1 for r in results if r['warnings'])

    print(f"\n{'='*70}")
    print(f"SUMMARY: {name}")
    print(f"{'='*70}")
    print(f"Total files:        {len(results)}")
    print(f"✅ Valid files:     {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"❌ Invalid files:   {invalid_count} ({invalid_count/len(results)*100:.1f}%)")
    print(f"⚠️  Files w/ warnings: {warning_count} ({warning_count/len(results)*100:.1f}%)")

    # Shape distribution
    if valid_count > 0:
        print(f"\nFeature shapes:")
        shapes = {}
        for r in results:
            if r['valid'] and r['features_shape']:
                shape_str = str(r['features_shape'])
                shapes[shape_str] = shapes.get(shape_str, 0) + 1

        for shape, count in sorted(shapes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {shape}: {count} files")

    # List invalid files
    if invalid_count > 0:
        print(f"\n❌ INVALID FILES ({invalid_count}):")
        for r in results:
            if not r['valid']:
                print(f"\n  {os.path.basename(r['path'])}:")
                for error in r['errors']:
                    print(f"    - {error}")

    # List files with warnings
    if warning_count > 0 and verbose:
        print(f"\n⚠️  FILES WITH WARNINGS ({warning_count}):")
        for r in results:
            if r['warnings'] and r['valid']:
                print(f"\n  {os.path.basename(r['path'])}:")
                for warning in r['warnings']:
                    print(f"    - {warning}")

    return results


def compare_directories(old_results, new_results):
    """Compare validation results from two directories"""

    print(f"\n{'='*70}")
    print("COMPARISON: OLD vs NEW")
    print(f"{'='*70}")

    if not old_results or not new_results:
        print("❌ Cannot compare - missing results from one or both directories")
        return

    # Compare counts
    print(f"\nFile counts:")
    print(f"  Old directory: {len(old_results)} files")
    print(f"  New directory: {len(new_results)} files")

    old_valid = sum(1 for r in old_results if r['valid'])
    new_valid = sum(1 for r in new_results if r['valid'])

    print(f"\nValid files:")
    print(f"  Old directory: {old_valid}/{len(old_results)} ({old_valid/len(old_results)*100:.1f}%)")
    print(f"  New directory: {new_valid}/{len(new_results)} ({new_valid/len(new_results)*100:.1f}%)")

    # Check if new has issues old doesn't
    if old_valid == len(old_results) and new_valid < len(new_results):
        print(f"\n⚠️  WARNING: Old directory has no invalid files, but new directory has {len(new_results) - new_valid} invalid files!")
        print("   This suggests an issue with the new embedding generation.")

    # Compare slide IDs
    old_slides = set(os.path.basename(r['path']).replace('.h5', '') for r in old_results)
    new_slides = set(os.path.basename(r['path']).replace('.h5', '') for r in new_results)

    common_slides = old_slides & new_slides
    old_only = old_slides - new_slides
    new_only = new_slides - old_slides

    print(f"\nSlide coverage:")
    print(f"  Common slides: {len(common_slides)}")
    print(f"  Old only:      {len(old_only)}")
    print(f"  New only:      {len(new_only)}")

    if old_only:
        print(f"\n  Examples of slides in OLD but not NEW (first 5):")
        for slide_id in list(old_only)[:5]:
            print(f"    - {slide_id}")

    if new_only:
        print(f"\n  Examples of slides in NEW but not OLD (first 5):")
        for slide_id in list(new_only)[:5]:
            print(f"    - {slide_id}")


def main():
    print("="*70)
    print("H5 EMBEDDING VALIDATION SCRIPT")
    print("="*70)

    # Check if directories exist
    print(f"\nChecking directories...")
    print(f"  Old: {OLD_FEATS_PATH}")
    print(f"    Exists: {os.path.exists(OLD_FEATS_PATH)}")
    print(f"  New: {NEW_FEATS_PATH}")
    print(f"    Exists: {os.path.exists(NEW_FEATS_PATH)}")

    if not os.path.exists(OLD_FEATS_PATH):
        print(f"\n❌ Old directory does not exist: {OLD_FEATS_PATH}")
        return

    if not os.path.exists(NEW_FEATS_PATH):
        print(f"\n❌ New directory does not exist: {NEW_FEATS_PATH}")
        return

    # Validate old directory (sample only for speed)
    old_results = validate_directory(OLD_FEATS_PATH, name="OLD (Working)", max_files=100, verbose=False)

    # Validate new directory (all files to find issues)
    new_results = validate_directory(NEW_FEATS_PATH, name="NEW (Testing)", max_files=None, verbose=True)

    # Compare
    if old_results and new_results:
        compare_directories(old_results, new_results)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
