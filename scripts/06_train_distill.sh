#!/usr/bin/env bash
# =============================================================================
# Step 5: Distill ZipVoice-ID to Fewer Inference Steps
# =============================================================================
# Distills the base flow-matching model to run with fewer NFE (function
# evaluations) at inference time. This is how LuxTTS achieves fast TTS:
# the base model needs many steps, but the distilled model converges in 4-8.
#
# Requires: a trained base model checkpoint from step 05_train.sh
#
# The distilled model is smaller/faster at inference but requires the full
# training process first.
# =============================================================================

set -euo pipefail

# --- Configuration ---
DATASET="indonesian"
MANIFEST_DIR="data/fbank"
TOKEN_FILE="data/tokens_id.txt"
MODEL_CONFIG="conf/zipvoice_base.json"
BASE_EXP_DIR="exp/zipvoice-id"       # Where base model checkpoints live
DISTILL_EXP_DIR="exp/zipvoice-id-distill"

# Training hyperparameters
WORLD_SIZE=1
NUM_EPOCHS=10       # Distillation usually needs fewer epochs than base training
MAX_DURATION=200
LANG="id"

# Select the best checkpoint from base training
# (adjust epoch number to your best checkpoint)
BASE_CHECKPOINT="$BASE_EXP_DIR/best-valid-loss.pt"
# Or use a specific epoch: BASE_CHECKPOINT="$BASE_EXP_DIR/epoch-20.pt"

# --- Validate checkpoint exists ---
if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "ERROR: Base checkpoint not found: $BASE_CHECKPOINT"
    echo "Run 05_train.sh first and adjust BASE_CHECKPOINT path."
    exit 1
fi

mkdir -p "$DISTILL_EXP_DIR"

echo "=== Distilling ZipVoice-ID ==="
echo "Base checkpoint: $BASE_CHECKPOINT"
echo "Distill exp dir: $DISTILL_EXP_DIR"
echo ""

python3 -m zipvoice.bin.train_zipvoice_distill \
    --world-size "$WORLD_SIZE" \
    --use-fp16 1 \
    --num-epochs "$NUM_EPOCHS" \
    --max-duration "$MAX_DURATION" \
    --model-config "$MODEL_CONFIG" \
    --tokenizer espeak \
    --lang "$LANG" \
    --token-file "$TOKEN_FILE" \
    --dataset custom \
    --manifest-dir "$MANIFEST_DIR" \
    --base-model "$BASE_CHECKPOINT" \
    --exp-dir "$DISTILL_EXP_DIR"

echo ""
echo "=== Distillation complete! Checkpoints in $DISTILL_EXP_DIR ==="
echo ""
echo "Next: run 07_inference.sh with --checkpoint $DISTILL_EXP_DIR/best-valid-loss.pt"
