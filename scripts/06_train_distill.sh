#!/usr/bin/env bash
# =============================================================================
# Step 5: Distill ZipVoice-ID to Fewer Inference Steps
# =============================================================================
# Distills the base flow-matching model so it runs with fewer ODE steps
# at inference (4-8 instead of 20-50). Two-stage process:
#
#   Stage 1 (first): Fixed teacher (base ZipVoice) → student (ZipVoiceDistill)
#     - Uses --num-iters (not epochs), typically 60K iters
#     - Only fm_decoder is trainable (text encoder frozen)
#
#   Stage 2 (second): EMA self-distillation
#     - Uses the stage 1 output as starting point
#     - Teacher = EMA of student (updated each step)
#     - Typically 2K iters, lower LR
#
# Requires: a trained base model checkpoint from step 05_train.sh
# =============================================================================

set -euo pipefail

# --- Configuration ---
DATASET="indonesian"
TOKEN_FILE="data/tokens_id.txt"
MODEL_CONFIG="conf/zipvoice_base.json"

# Manifest paths (same as training)
TRAIN_MANIFEST="data/fbank/${DATASET}_cuts_train_tokens.jsonl.gz"
DEV_MANIFEST="data/fbank/${DATASET}_cuts_valid_tokens.jsonl.gz"

# Base model checkpoint (from step 05)
# Use the averaged model for best results:
TEACHER_MODEL="exp/zipvoice-id/best-valid-loss.pt"

# Distillation experiment directories
STAGE1_EXP_DIR="exp/zipvoice-id-distill-stage1"
STAGE2_EXP_DIR="exp/zipvoice-id-distill"

# Training hyperparameters
WORLD_SIZE=1
MAX_DURATION=200
LANG="id"

# --- Validate teacher checkpoint ---
if [ ! -f "$TEACHER_MODEL" ]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER_MODEL"
    echo "Run 05_train.sh first and adjust TEACHER_MODEL path."
    echo ""
    echo "Available checkpoints:"
    ls -lh exp/zipvoice-id/*.pt 2>/dev/null || echo "  (none found)"
    exit 1
fi

# =============================================================================
# Stage 1: Fixed teacher distillation
# =============================================================================
echo "=== Stage 1: Fixed teacher distillation ==="
echo "Teacher: $TEACHER_MODEL"
echo "Exp dir: $STAGE1_EXP_DIR"
echo ""

mkdir -p "$STAGE1_EXP_DIR"

python3 -m zipvoice.bin.train_zipvoice_distill \
    --world-size "$WORLD_SIZE" \
    --use-fp16 1 \
    --num-iters 60000 \
    --save-every-n 5000 \
    --max-duration "$MAX_DURATION" \
    --base-lr 0.0005 \
    --model-config "$MODEL_CONFIG" \
    --tokenizer espeak \
    --lang "$LANG" \
    --token-file "$TOKEN_FILE" \
    --dataset custom \
    --train-manifest "$TRAIN_MANIFEST" \
    --dev-manifest "$DEV_MANIFEST" \
    --teacher-model "$TEACHER_MODEL" \
    --distill-stage first \
    --exp-dir "$STAGE1_EXP_DIR"

echo ""
echo "=== Stage 1 complete! ==="

# =============================================================================
# Stage 2: EMA self-distillation
# =============================================================================

# Use the best or last checkpoint from stage 1
# ZipVoice uses generate_averaged_model.py to average checkpoints.
# For simplicity, use the last iter checkpoint:
STAGE1_CKPT="$STAGE1_EXP_DIR/checkpoint-60000.pt"
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "WARNING: Expected $STAGE1_CKPT not found."
    echo "Looking for latest checkpoint..."
    STAGE1_CKPT=$(ls -t "$STAGE1_EXP_DIR"/checkpoint-*.pt 2>/dev/null | head -1)
    if [ -z "$STAGE1_CKPT" ]; then
        echo "ERROR: No stage 1 checkpoints found. Skipping stage 2."
        exit 1
    fi
    echo "Using: $STAGE1_CKPT"
fi

echo ""
echo "=== Stage 2: EMA self-distillation ==="
echo "Teacher: $STAGE1_CKPT"
echo "Exp dir: $STAGE2_EXP_DIR"
echo ""

mkdir -p "$STAGE2_EXP_DIR"

python3 -m zipvoice.bin.train_zipvoice_distill \
    --world-size "$WORLD_SIZE" \
    --use-fp16 1 \
    --num-iters 2000 \
    --save-every-n 1000 \
    --max-duration "$MAX_DURATION" \
    --base-lr 0.0001 \
    --model-config "$MODEL_CONFIG" \
    --tokenizer espeak \
    --lang "$LANG" \
    --token-file "$TOKEN_FILE" \
    --dataset custom \
    --train-manifest "$TRAIN_MANIFEST" \
    --dev-manifest "$DEV_MANIFEST" \
    --teacher-model "$STAGE1_CKPT" \
    --distill-stage second \
    --ema-decay 0.9999 \
    --exp-dir "$STAGE2_EXP_DIR"

echo ""
echo "=== Distillation complete! ==="
echo "Final model: $STAGE2_EXP_DIR"
echo ""
echo "Next: run 07_inference.sh with the distilled checkpoint."
echo "  Distilled model uses 4-8 steps (vs 20-50 for base model)."
