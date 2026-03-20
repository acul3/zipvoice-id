#!/usr/bin/env bash
# =============================================================================
# Step 2: Compute Mel Filterbank Features (fbank)
# =============================================================================
# Extracts 100-dim mel filterbank features from audio and saves them as
# Lhotse CutSets with features attached.
#
# Input:  data/manifests/{dataset}_cuts_{subset}.jsonl.gz
# Output: data/fbank/{dataset}_cuts_{subset}.jsonl.gz  (with fbank arrays)
#
# Audio is resampled to 24kHz during this step.
# This step can be slow — set NUM_JOBS based on your CPU cores.
# =============================================================================

set -euo pipefail

# --- Configuration ---
DATASET="indonesian"
SUBSET="train"            # train | valid | test
SOURCE_DIR="data/manifests"
DEST_DIR="data/fbank"
NUM_JOBS=4                # Parallel workers (set to nproc for max speed)

# --- Create output directory ---
mkdir -p "$DEST_DIR"

echo "=== Computing fbank for $DATASET/$SUBSET ==="
python3 -m zipvoice.bin.compute_fbank \
    --dataset "$DATASET" \
    --subset "$SUBSET" \
    --source-dir "$SOURCE_DIR" \
    --dest-dir "$DEST_DIR" \
    --num-jobs "$NUM_JOBS"

# Repeat for valid/test if needed:
# python3 -m zipvoice.bin.compute_fbank \
#     --dataset "$DATASET" --subset valid \
#     --source-dir "$SOURCE_DIR" --dest-dir "$DEST_DIR" --num-jobs "$NUM_JOBS"

echo "=== Done! Features saved to $DEST_DIR ==="
ls -lh "$DEST_DIR"
