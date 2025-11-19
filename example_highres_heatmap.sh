#!/bin/bash
# Example: Generate high-resolution heatmap with 90% overlap

# Basic usage - 90% overlap at full resolution
# Note: Feature extractor is AUTO-DETECTED from checkpoint!
# Replace SLIDE_ID with actual slide ID
python generate_heatmaps_highres.py \
    --checkpoint best_model_abmil.pth \
    --slide /media/nadim/Data/prostate-cancer-grade-assessment/train_images/SLIDE_ID.tiff \
    --output heatmaps/highres/SLIDE_ID_heatmap_90overlap.tiff \
    --overlap 0.90

# Ultra high-resolution - 95% overlap
python generate_heatmaps_highres.py \
    --checkpoint best_model_abmil.pth \
    --slide /media/nadim/Data/prostate-cancer-grade-assessment/train_images/SLIDE_ID.tiff \
    --output heatmaps/highres/SLIDE_ID_heatmap_95overlap.tiff \
    --overlap 0.95 \
    --vis_level 0

# Slightly lower resolution - faster processing
python generate_heatmaps_highres.py \
    --checkpoint best_model_abmil.pth \
    --slide /media/nadim/Data/prostate-cancer-grade-assessment/train_images/SLIDE_ID.tiff \
    --output heatmaps/highres/SLIDE_ID_heatmap_90overlap_2x.tiff \
    --overlap 0.90 \
    --vis_level 1  # 2x downsampled

# With custom performance settings
python generate_heatmaps_highres.py \
    --checkpoint best_model_abmil.pth \
    --slide /media/nadim/Data/prostate-cancer-grade-assessment/train_images/SLIDE_ID.tiff \
    --output heatmaps/highres/SLIDE_ID_heatmap.tiff \
    --overlap 0.90 \
    --vis_level 0 \
    --batch_size_extract 16 \
    --batch_size_infer 1024

# Different colormap and opacity
python generate_heatmaps_highres.py \
    --checkpoint best_model_abmil.pth \
    --slide /media/nadim/Data/prostate-cancer-grade-assessment/train_images/SLIDE_ID.tiff \
    --output heatmaps/highres/SLIDE_ID_heatmap.tiff \
    --overlap 0.90 \
    --cmap plasma \
    --alpha 0.6
