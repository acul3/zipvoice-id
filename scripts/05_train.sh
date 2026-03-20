#!/usr/bin/env bash
# =============================================================================
# Step 4: Train ZipVoice-ID (Base Model)
# =============================================================================
# Trains the 123M parameter ZipVoice flow-matching TTS model from scratch
# on your Indonesian dataset.
#
# IMPORTANT: --dataset custom requires --train-manifest and --dev-manifest
# (not auto-discovered from --manifest-dir)
#
# Recommended hardware: GPU with 10GB+ VRAM
# Jetson AGX Orin (64GB unified memory): runs well with max-duration=150-200
#
# Training time estimates (single GPU):
#   10h dataset  → ~12-24h to convergence
#   100h dataset → ~3-7 days to convergence
#
# Checkpoints are saved to exp-dir every save-every-n batches.
# Monitor with: tensorboard --logdir exp/zipvoice-id
# =============================================================================

set -euo pipefail

# --- Configuration ---
DATASET="indonesian"
TOKEN_FILE="data/tokens_id.txt"
MODEL_CONFIG="conf/zipvoice_base.json"
EXP_DIR="exp/zipvoice-id"

# Manifest paths (output of step 03 — tokenized fbank manifests)
# These must be the TOKENIZED manifests (from prepare_tokens), not raw manifests.
# If you skipped step 03, the training script will tokenize on-the-fly (slower).
TRAIN_MANIFEST="data/fbank/${DATASET}_cuts_train_tokens.jsonl.gz"
DEV_MANIFEST="data/fbank/${DATASET}_cuts_valid_tokens.jsonl.gz"

# Training hyperparameters
WORLD_SIZE=1        # Number of GPUs (1 for Jetson)
NUM_EPOCHS=20       # Training epochs
MAX_DURATION=200    # Max batch duration in seconds (lower if OOM: try 100-150)
LR_HOURS=5000       # Learning rate schedule horizon in data-hours
BASE_LR=0.02        # Base learning rate (ZipVoice default)
SAVE_EVERY_N=5000   # Save checkpoint every N batches
NUM_BUCKETS=30      # Buckets for dynamic batching (must be <= num training cuts)
LANG="id"

# --- Validate inputs ---
if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $TOKEN_FILE"
    echo "Run tools/build_tokenizer.py first (see Step 0 in README)."
    exit 1
fi

if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "ERROR: Training manifest not found: $TRAIN_MANIFEST"
    echo "Run scripts 01-03 first, or adjust TRAIN_MANIFEST path."
    echo ""
    echo "If you have non-tokenized manifests, you can use them too —"
    echo "the trainer will tokenize on-the-fly (slower)."
    echo "  TRAIN_MANIFEST=data/fbank/${DATASET}_cuts_train.jsonl.gz"
    exit 1
fi

if [ ! -f "$DEV_MANIFEST" ]; then
    echo "WARNING: Dev manifest not found: $DEV_MANIFEST"
    echo "Training will fail without a validation set."
    echo "Prepare one with scripts/01_prepare_dataset.sh (subset=valid)."
fi

# --- Create experiment directory ---
mkdir -p "$EXP_DIR"

echo "=== Training ZipVoice-ID ==="
echo "Train manifest: $TRAIN_MANIFEST"
echo "Dev manifest:   $DEV_MANIFEST"
echo "Token file:     $TOKEN_FILE"
echo "Model config:   $MODEL_CONFIG"
echo "Exp dir:        $EXP_DIR"
echo "Max duration:   $MAX_DURATION s"
echo "Epochs:         $NUM_EPOCHS"
echo ""

python3 -m zipvoice.bin.train_zipvoice \
    --world-size "$WORLD_SIZE" \
    --use-fp16 1 \
    --num-epochs "$NUM_EPOCHS" \
    --max-duration "$MAX_DURATION" \
    --lr-hours "$LR_HOURS" \
    --base-lr "$BASE_LR" \
    --save-every-n "$SAVE_EVERY_N" \
    --num-buckets "$NUM_BUCKETS" \
    --model-config "$MODEL_CONFIG" \
    --tokenizer espeak \
    --lang "$LANG" \
    --token-file "$TOKEN_FILE" \
    --dataset custom \
    --train-manifest "$TRAIN_MANIFEST" \
    --dev-manifest "$DEV_MANIFEST" \
    --exp-dir "$EXP_DIR"

echo ""
echo "=== Training complete! ==="
echo "Checkpoints in: $EXP_DIR"
echo "Best model:     $EXP_DIR/best-valid-loss.pt"

# Tips for Jetson AGX Orin:
# - If OOM: reduce --max-duration to 100-150
# - Use --use-fp16 1 (already set) for memory savings
# - Monitor GPU: sudo tegrastats
# - Check temps: sudo cat /sys/devices/virtual/thermal/thermal_zone*/temp
# - Resume from checkpoint: add --start-epoch N (loads epoch-(N-1).pt)
