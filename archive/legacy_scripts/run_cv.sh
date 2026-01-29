#!/bin/bash
# End-to-end K-fold Cross-Validation

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
LABELS_CSV="panda_labels.csv"
FEATURES_DIR="/media/nadim/Data/Imaging/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/"

# Model settings
MODEL="abmil.base.uni_v2.pc108-24k"
NUM_CLASSES=6
NUM_HEADS=1

# CV settings
N_FOLDS=5
VAL_RATIO=0.1
SEED=42

# Training settings
EPOCHS=20
LEARNING_RATE=1e-4
EARLY_STOPPING_PATIENCE=10
MIN_EPOCHS=5
TASK_TYPE="multiclass"  # "binary" or "multiclass"
EARLY_STOPPING_METRIC="kappa"  # "auto", "kappa", "balanced_accuracy", "auc"

# Multi-slide settings
GROUP_COLUMN="case_id"
HIERARCHICAL=false  # true for late fusion, false for early fusion

# Output settings
OUTPUT_DIR="experiments/cv_run"
WORKERS=1  # Number of parallel fold workers

# =============================================================================
# DERIVED PATHS
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
SPLIT_DIR="${RUN_DIR}/splits"
CONFIG_FILE="${RUN_DIR}/config.json"

# =============================================================================
# STEP 1: Create output directory
# =============================================================================

echo "============================================================"
echo "K-FOLD CROSS-VALIDATION"
echo "============================================================"
echo ""
echo "Model:      $MODEL"
echo "Folds:      $N_FOLDS"
echo "Output:     $RUN_DIR"
echo ""

mkdir -p "$RUN_DIR"

# =============================================================================
# STEP 2: Generate splits
# =============================================================================

echo "------------------------------------------------------------"
echo "STEP 1: Generating $N_FOLDS-fold splits..."
echo "------------------------------------------------------------"

python scripts/create_splits.py \
    --label_csv "$LABELS_CSV" \
    --output_dir "$SPLIT_DIR" \
    --n_folds "$N_FOLDS" \
    --val_ratio "$VAL_RATIO" \
    --seed "$SEED"

echo ""

# =============================================================================
# STEP 3: Generate config JSON
# =============================================================================

echo "------------------------------------------------------------"
echo "STEP 2: Generating config..."
echo "------------------------------------------------------------"

cat > "$CONFIG_FILE" << EOF
{
    "data": {
        "labels_csv": "$LABELS_CSV",
        "features_dir": "$FEATURES_DIR",
        "group_column": "$GROUP_COLUMN",
        "hierarchical": $HIERARCHICAL
    },
    "train": {
        "num_epochs": $EPOCHS,
        "learning_rate": $LEARNING_RATE,
        "early_stopping_patience": $EARLY_STOPPING_PATIENCE,
        "min_epochs": $MIN_EPOCHS,
        "task_type": "$TASK_TYPE",
        "early_stopping_metric": "$EARLY_STOPPING_METRIC"
    },
    "model_name": "$MODEL",
    "num_classes": $NUM_CLASSES,
    "num_heads": $NUM_HEADS,
    "output_dir": "$RUN_DIR"
}
EOF

echo "Config saved to: $CONFIG_FILE"
echo ""

# =============================================================================
# STEP 4: Run CV orchestrator
# =============================================================================

echo "------------------------------------------------------------"
echo "STEP 3: Running cross-validation..."
echo "------------------------------------------------------------"

python scripts/run_cv_orchestrator.py \
    --config "$CONFIG_FILE" \
    --split_dir "$SPLIT_DIR" \
    --output_dir "$RUN_DIR" \
    --workers "$WORKERS"

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Results saved to: $RUN_DIR"
echo "Summary:          $RUN_DIR/cv_summary.csv"
echo ""
