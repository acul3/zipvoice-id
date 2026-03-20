#!/usr/bin/env bash
# =============================================================================
# Step 6: Inference with ZipVoice-ID
# =============================================================================
# Generate speech from text using your trained (or distilled) model.
# You need a reference audio clip (3-10 seconds) for voice cloning.
#
# Two inference modes:
#   1. Standard ZipVoice inference (24kHz output)
#   2. LuxTTS-style inference (48kHz via LinaCodec vocoder) — see notes below
# =============================================================================

set -euo pipefail

# --- Configuration ---
CHECKPOINT="exp/zipvoice-id-distill/best-valid-loss.pt"
# Or base model: CHECKPOINT="exp/zipvoice-id/best-valid-loss.pt"

TOKEN_FILE="data/tokens_id.txt"
MODEL_CONFIG="conf/zipvoice_base.json"

# Input
TEXT="Selamat datang di sistem text-to-speech bahasa Indonesia."
REFERENCE_AUDIO="data/reference/speaker_ref.wav"  # 3-10 sec reference clip
OUTPUT_WAV="output/generated.wav"

# Inference steps (fewer = faster, more = better quality)
# Distilled model: 4-8 steps is usually sufficient
# Base model: 20-50 steps needed
NUM_STEPS=8
LANG="id"

# --- Create output directory ---
mkdir -p "$(dirname "$OUTPUT_WAV")"

echo "=== ZipVoice-ID Inference ==="
echo "Text:      $TEXT"
echo "Reference: $REFERENCE_AUDIO"
echo "Output:    $OUTPUT_WAV"
echo "Steps:     $NUM_STEPS"
echo ""

python3 -m zipvoice.bin.infer \
    --checkpoint "$CHECKPOINT" \
    --model-config "$MODEL_CONFIG" \
    --token-file "$TOKEN_FILE" \
    --tokenizer espeak \
    --lang "$LANG" \
    --text "$TEXT" \
    --reference-audio "$REFERENCE_AUDIO" \
    --output "$OUTPUT_WAV" \
    --num-steps "$NUM_STEPS"

echo "=== Done! Output saved to $OUTPUT_WAV ==="

# =============================================================================
# LuxTTS-Style 48kHz Inference (Optional)
# =============================================================================
# To get 48kHz output using LinaCodec vocoder (like LuxTTS), use a Python
# script that:
#   1. Runs ZipVoice to get 24kHz mel features (not decoded audio)
#   2. Passes mel features through LinaCodec decoder for 48kHz output
#   3. Uses anchor sampler for faster/better distilled model inference
#
# See README.md "LuxTTS-Style Inference" section for full Python example.
# =============================================================================
