#!/usr/bin/env bash
# =============================================================================
# Step 1: Prepare Dataset — TSV → Lhotse Manifests
# =============================================================================
# Converts your TSV-formatted dataset into Lhotse CutSet manifests that
# ZipVoice expects. Each line in the TSV should be:
#   {id}\t{text}\t{wav_path}
# or with timestamps:
#   {id}\t{text}\t{wav_path}\t{start}\t{end}
#
# Audio can be any sample rate; it will be resampled to 24kHz during fbank.
# =============================================================================

set -euo pipefail

# --- Configuration ---
DATASET="indonesian"              # Name prefix for output files
TSV_PATH="data/raw/indonesian_train.tsv"   # Path to your training TSV
OUTPUT_DIR="data/manifests"       # Where to save lhotse manifests
NUM_JOBS=4                        # Parallel workers

# Optional: validation/test split TSVs
TSV_VALID="data/raw/indonesian_valid.tsv"
TSV_TEST="data/raw/indonesian_test.tsv"

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

echo "=== Preparing training split ==="
python3 -m zipvoice.bin.prepare_dataset \
    --tsv-path "$TSV_PATH" \
    --prefix "$DATASET" \
    --subset train \
    --output-dir "$OUTPUT_DIR" \
    --num-jobs "$NUM_JOBS"

# Uncomment to also prepare validation/test splits:
# echo "=== Preparing validation split ==="
# python3 -m zipvoice.bin.prepare_dataset \
#     --tsv-path "$TSV_VALID" \
#     --prefix "$DATASET" \
#     --subset valid \
#     --output-dir "$OUTPUT_DIR" \
#     --num-jobs "$NUM_JOBS"

# echo "=== Preparing test split ==="
# python3 -m zipvoice.bin.prepare_dataset \
#     --tsv-path "$TSV_TEST" \
#     --prefix "$DATASET" \
#     --subset test \
#     --output-dir "$OUTPUT_DIR" \
#     --num-jobs "$NUM_JOBS"

echo "=== Done! Manifests saved to $OUTPUT_DIR ==="
ls -lh "$OUTPUT_DIR"
