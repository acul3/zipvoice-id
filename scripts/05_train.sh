#!/usr/bin/env bash
# =============================================================================
# Step 4: Train ZipVoice-ID (Base Model)
# =============================================================================
# Trains the 123M parameter ZipVoice flow-matching TTS model from scratch
# on your Indonesian dataset.
#
# Recommended hardware: GPU with 10GB+ VRAM
# Jetson AGX Orin (64GB unified memory): runs well with max-duration=150-200
#
# Training time estimates (single GPU):
#   10h dataset  → ~12-24h to convergence
#   100h dataset → ~3-7 days to convergence
#
# Checkpoints are saved to exp-dir every epoch.
# Monitor with: tensorboard --logdir exp/zipvoice-id
# =============================================================================

set -euo pipefail

# --- Configuration ---
DATASET="indonesian"
MANIFEST_DIR="data/fbank"
TOKEN_FILE="data/tokens_id.txt"
MODEL_CONFIG="conf/zipvoice_base.json"
EXP_DIR="exp/zipvoice-id"

# Training hyperparameters
WORLD_SIZE=1        # Number of GPUs (1 for Jetson)
NUM_EPOCHS=20       # Training epochs
MAX_DURATION=200    # Max batch duration in seconds (lower if OOM)
LR_HOURS=5000       # Learning rate schedule horizon in data-hours
LANG="id"

# --- Create experiment directory ---
mkdir -p "$EXP_DIR"

echo "=== Training ZipVoice-ID ==="
echo "Dataset:    $DATASET"
echo "Manifest:   $MANIFEST_DIR"
echo "Token file: $TOKEN_FILE"
echo "Exp dir:    $EXP_DIR"
echo ""

python3 -m zipvoice.bin.train_zipvoice \
    --world-size "$WORLD_SIZE" \
    --use-fp16 1 \
    --num-epochs "$NUM_EPOCHS" \
    --max-duration "$MAX_DURATION" \
    --lr-hours "$LR_HOURS" \
    --model-config "$MODEL_CONFIG" \
    --tokenizer espeak \
    --lang "$LANG" \
    --token-file "$TOKEN_FILE" \
    --dataset custom \
    --manifest-dir "$MANIFEST_DIR" \
    --exp-dir "$EXP_DIR"

echo ""
echo "=== Training complete! Checkpoints in $EXP_DIR ==="

# Tips for Jetson AGX Orin:
# - If OOM: reduce --max-duration to 100-150
# - Use --use-fp16 1 (already set) for memory savings
# - Monitor GPU: sudo tegrastats
# - Check temps: sudo cat /sys/devices/virtual/thermal/thermal_zone*/temp
