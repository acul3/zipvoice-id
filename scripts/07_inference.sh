#!/usr/bin/env bash
# =============================================================================
# Step 6: Inference with ZipVoice-ID
# =============================================================================
# Generate speech from text using your trained (or distilled) model.
# You need a reference audio clip (3-10 seconds) for voice cloning.
#
# ZipVoice inference expects a model directory containing:
#   model.json   — model config (copy from conf/zipvoice_base.json)
#   tokens.txt   — phoneme vocab (copy from data/tokens_id.txt)
#   model.pt     — checkpoint (or rename your checkpoint)
#
# The training script auto-copies model.json and tokens.txt to exp-dir.
# You just need to rename/symlink your best checkpoint to model.pt.
# =============================================================================

set -euo pipefail

# --- Configuration ---
# Model directory (training auto-creates model.json + tokens.txt here)
MODEL_DIR="exp/zipvoice-id"

# Which checkpoint to use (will be symlinked as model.pt)
CHECKPOINT="best-valid-loss.pt"

# Model type: "zipvoice" (base, 20-50 steps) or "zipvoice_distill" (4-8 steps)
MODEL_NAME="zipvoice"
NUM_STEPS=30

# For distilled model, uncomment:
# MODEL_DIR="exp/zipvoice-id-distill"
# CHECKPOINT="checkpoint-2000.pt"
# MODEL_NAME="zipvoice_distill"
# NUM_STEPS=8

# Input
PROMPT_WAV="data/reference/speaker_ref.wav"    # 3-10 sec reference clip
PROMPT_TEXT="Transkrip dari audio referensi."   # Transcript of reference audio
TEXT="Selamat datang di sistem text-to-speech bahasa Indonesia."
OUTPUT_WAV="output/generated.wav"

LANG="id"

# --- Validate ---
if [ ! -f "$MODEL_DIR/$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $MODEL_DIR/$CHECKPOINT"
    echo "Train a model first (scripts 05-06)."
    echo ""
    echo "Available checkpoints:"
    ls -lh "$MODEL_DIR"/*.pt 2>/dev/null || echo "  (none found)"
    exit 1
fi

if [ ! -f "$PROMPT_WAV" ]; then
    echo "ERROR: Reference audio not found: $PROMPT_WAV"
    echo "Provide a 3-10 second WAV file of the target speaker."
    exit 1
fi

# --- Prepare model directory ---
# The training script copies model.json and tokens.txt to exp-dir.
# Verify they exist:
if [ ! -f "$MODEL_DIR/model.json" ]; then
    echo "WARNING: model.json not found in $MODEL_DIR"
    echo "Copying from conf/zipvoice_base.json..."
    cp conf/zipvoice_base.json "$MODEL_DIR/model.json"
fi

if [ ! -f "$MODEL_DIR/tokens.txt" ]; then
    echo "WARNING: tokens.txt not found in $MODEL_DIR"
    echo "Copying from data/tokens_id.txt..."
    cp data/tokens_id.txt "$MODEL_DIR/tokens.txt"
fi

# Symlink checkpoint as model.pt (if not already there)
if [ ! -f "$MODEL_DIR/model.pt" ] || [ "$(readlink -f "$MODEL_DIR/model.pt")" != "$(readlink -f "$MODEL_DIR/$CHECKPOINT")" ]; then
    echo "Symlinking $CHECKPOINT → model.pt"
    ln -sf "$CHECKPOINT" "$MODEL_DIR/model.pt"
fi

# --- Create output directory ---
mkdir -p "$(dirname "$OUTPUT_WAV")"

echo "=== ZipVoice-ID Inference ==="
echo "Model dir:  $MODEL_DIR"
echo "Model type: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Text:       $TEXT"
echo "Reference:  $PROMPT_WAV"
echo "Output:     $OUTPUT_WAV"
echo "Steps:      $NUM_STEPS"
echo ""

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name "$MODEL_NAME" \
    --model-dir "$MODEL_DIR" \
    --tokenizer espeak \
    --lang "$LANG" \
    --prompt-wav "$PROMPT_WAV" \
    --prompt-text "$PROMPT_TEXT" \
    --text "$TEXT" \
    --res-wav-path "$OUTPUT_WAV" \
    --num-step "$NUM_STEPS"

echo ""
echo "=== Done! Output saved to $OUTPUT_WAV ==="

# =============================================================================
# Batch inference from TSV
# =============================================================================
# Create test.tsv with format:
#   {wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}
#
# python3 -m zipvoice.bin.infer_zipvoice \
#     --model-name zipvoice_distill \
#     --model-dir exp/zipvoice-id-distill \
#     --tokenizer espeak \
#     --lang id \
#     --test-list test.tsv \
#     --res-dir output/batch/ \
#     --num-step 8

# =============================================================================
# LuxTTS-Style 48kHz Inference (Python)
# =============================================================================
# pip install git+https://github.com/ysharma3501/LinaCodec.git
#
# Then package your model dir with LinaCodec's vocoder:
#   cp -r exp/zipvoice-id-distill exp/zipvoice-id-lux
#   # Download vocoder from LuxTTS HuggingFace
#   huggingface-cli download YatharthS/LuxTTS vocoder/ --local-dir exp/zipvoice-id-lux
#
# Python:
#   from zipvoice.luxvoice import LuxTTS
#   lux = LuxTTS(model_path="exp/zipvoice-id-lux", device="cuda")
#   encoded = lux.encode_prompt("reference.wav", duration=5, rms=0.01)
#   wav = lux.generate_speech("Halo, apa kabar?", encoded, num_steps=4)
#   import soundfile as sf
#   sf.write("output_48k.wav", wav.numpy().squeeze(), 48000)
