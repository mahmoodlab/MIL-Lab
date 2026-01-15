#!/bin/bash
# Example: Generate heatmaps for PANDA dataset with trained ABMIL model

# Simple usage - just provide 4 paths:
python generate_heatmaps.py \
    --checkpoint /path/to/your/abmil_checkpoint.pt \
    --h5_dir /media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/ \
    --slide_dir /media/nadim/Data/prostate-cancer-grade-assessment/train_images \
    --output_dir heatmaps/panda_output

# With custom visualization settings:
python generate_heatmaps.py \
    --checkpoint /path/to/your/abmil_checkpoint.pt \
    --h5_dir /media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/ \
    --slide_dir /media/nadim/Data/prostate-cancer-grade-assessment/train_images \
    --output_dir heatmaps/panda_output \
    --cmap plasma \
    --alpha 0.5 \
    --slide_ext .tiff

# Process only specific slides from CSV:
python generate_heatmaps.py \
    --checkpoint /path/to/your/abmil_checkpoint.pt \
    --h5_dir /media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/ \
    --slide_dir /media/nadim/Data/prostate-cancer-grade-assessment/train_images \
    --output_dir heatmaps/panda_output \
    --csv test_slides.csv

# Process just first 10 slides for quick test:
python generate_heatmaps.py \
    --checkpoint /path/to/your/abmil_checkpoint.pt \
    --h5_dir /media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/ \
    --slide_dir /media/nadim/Data/prostate-cancer-grade-assessment/train_images \
    --output_dir heatmaps/panda_test \
    --limit 10
