#!/usr/bin/env bash
# =============================================================================
# Step 3: Tokenize Text with Espeak (Indonesian)
# =============================================================================
# Runs espeak-ng phonemization on all utterances in the fbank manifest.
# Uses the 'id' locale for Indonesian. Produces a new manifest with
# phoneme token sequences attached to each cut.
#
# Input:  data/fbank/{dataset}_cuts_{subset}.jsonl.gz
# Output: data/fbank/{dataset}_cuts_{subset}_tokens.jsonl.gz
#
# Requires: espeak-ng installed (sudo apt install espeak-ng)
# =============================================================================

set -euo pipefail

# --- Configuration ---
DATASET="indonesian"
SUBSET="train"
FBANK_DIR="data/fbank"
LANG="id"          # Indonesian espeak locale
NUM_JOBS=4

INPUT_FILE="$FBANK_DIR/${DATASET}_cuts_${SUBSET}.jsonl.gz"
OUTPUT_FILE="$FBANK_DIR/${DATASET}_cuts_${SUBSET}_tokens.jsonl.gz"

echo "=== Tokenizing $INPUT_FILE with espeak lang=$LANG ==="
python3 -m zipvoice.bin.prepare_tokens \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --tokenizer espeak \
    --lang "$LANG" \
    --num-jobs "$NUM_JOBS"

# Repeat for valid/test:
# python3 -m zipvoice.bin.prepare_tokens \
#     --input-file "$FBANK_DIR/${DATASET}_cuts_valid.jsonl.gz" \
#     --output-file "$FBANK_DIR/${DATASET}_cuts_valid_tokens.jsonl.gz" \
#     --tokenizer espeak --lang "$LANG" --num-jobs "$NUM_JOBS"

echo "=== Done! Tokenized manifest: $OUTPUT_FILE ==="
