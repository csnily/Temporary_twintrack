#!/bin/bash
# One-click experiment script for TwinTrack
# Usage: bash scripts/run_experiment.sh [dataset] [backbone]
set -e
DATASET=${1:-DCT}
BACKBONE=${2:-resnet101}

# Train
echo "[1/3] Training on $DATASET with $BACKBONE..."
python twintrack/train.py --dataset $DATASET --backbone $BACKBONE

# Test
echo "[2/3] Testing on $DATASET with $BACKBONE..."
python twintrack/test.py --dataset $DATASET --backbone $BACKBONE

# Visualize
echo "[3/3] Visualizing Grad-CAM on $DATASET with $BACKBONE..."
python twintrack/visualize.py --dataset $DATASET --backbone $BACKBONE --mode gradcam

echo "Experiment completed." 