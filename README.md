# ZipVoice-ID: Indonesian Text-to-Speech Training Recipe

Training recipe for an Indonesian TTS model based on the [ZipVoice](https://github.com/nickolay-kondratyev/ZipVoice) architecture — the same architecture behind [LuxTTS](https://huggingface.co/spaces/luxtts). This repo contains **only the training recipe** (configs, scripts, docs); the actual model code comes from the ZipVoice package.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Data Preparation](#data-preparation)
5. [Tokenizer](#tokenizer)
6. [Training Pipeline](#training-pipeline)
7. [Training on Jetson AGX Orin](#training-on-jetson-agx-orin)
8. [Distillation](#distillation)
9. [Inference](#inference)
10. [LuxTTS-Style Inference (48kHz)](#luxtts-style-inference-48khz)
11. [Estimated Timeline](#estimated-timeline)
12. [Adapting to Other Languages](#adapting-to-other-languages)

---

## Overview

ZipVoice-ID trains a **123M parameter flow-matching TTS model** for Indonesian (Bahasa Indonesia). Given text and a short reference audio clip (3–10 seconds), it generates natural-sounding speech that clones the reference speaker's voice.

Key features:
- **Zero-shot voice cloning** — any speaker with a short reference clip
- **Flow matching** — modern generative approach, better than diffusion for TTS
- **Distillation** — reduce inference from 50+ steps down to 4–8 steps
- **24kHz native** output, upgradable to **48kHz** with LinaCodec vocoder

## Architecture

The model has two main components:

### Text Encoder (Small, ~5M params)
- 4-layer Zipformer with 192-dim embeddings
- Converts phoneme tokens → text representations
- Config: `text_encoder_dim=192`, `text_encoder_num_layers=4`

### Flow-Matching Decoder (Large, ~118M params)
- 5-stack TTSZipformer with U-Net-like structure
- Downsampling factors: `[1, 2, 4, 2, 1]` (bottleneck at 4x)
- Layer counts: `[2, 2, 4, 4, 4]` = 16 total layers
- Decoder dim: 512, feedforward dim: 1536
- Takes text encodings + reference speaker embedding → generates 100-dim mel features
- Uses conditional flow matching (OT-CFM) for training

The total model is ~123M parameters, producing 100-dim Vocos-compatible mel spectrograms at 24kHz.

## Prerequisites

### Software

```bash
# Python 3.10+
python3 --version  # Should be 3.10+

# Install ZipVoice (the model code)
git clone https://github.com/nickolay-kondratyev/ZipVoice.git
cd ZipVoice && pip install -e . && cd ..

# Install espeak-ng (phonemizer backend)
sudo apt install espeak-ng

# Install this recipe's dependencies
cd zipvoice-id
pip install -r requirements.txt
```

### Hardware

- **GPU**: Any CUDA GPU with 8GB+ VRAM (10GB+ recommended)
- **Jetson AGX Orin**: Works great with 64GB unified memory
- **Storage**: ~50GB for 100h dataset (raw audio + features)
- **RAM**: 16GB+ recommended

### Verify Installation

```bash
# Test ZipVoice is importable
python3 -c "import zipvoice; print('ZipVoice OK')"

# Test espeak-ng
espeak-ng --voices | grep Indonesian

# Test piper_phonemize
python3 -c "from piper_phonemize import phonemize_espeak; print(phonemize_espeak('Halo dunia', 'id'))"
```

## Data Preparation

### Data Format

Prepare your data as TSV files with tab-separated columns:

```
{utterance_id}\t{text}\t{wav_path}
```

Or with timestamps for long audio files:

```
{utterance_id}\t{text}\t{wav_path}\t{start_seconds}\t{end_seconds}
```

Example:
```
indo_001	Selamat pagi, apa kabar?	wavs/speaker01/001.wav
indo_002	Terima kasih banyak.	wavs/speaker01/002.wav
indo_003	Saya pergi ke pasar.	wavs/speaker02/001.wav	0.0	3.5
```

### Audio Requirements

- **Format**: WAV (any sample rate — will be resampled to 24kHz)
- **Quality**: Clean recordings, minimal background noise preferred
- **Duration**: Individual utterances 1–30 seconds each
- **Speakers**: Multi-speaker is fine and encouraged for voice cloning

### Recommended Dataset Sizes

| Size | Quality | Notes |
|------|---------|-------|
| 1–5 hours | Proof of concept | Will sound robotic, good for testing pipeline |
| 10–30 hours | Baseline | Understandable but not great naturalness |
| 50–100 hours | Good | Natural prosody, decent voice cloning |
| 100+ hours | Best | High quality, robust voice cloning |

### Indonesian Speech Datasets

| Dataset | Hours | Link |
|---------|-------|------|
| Common Voice Indonesian | ~30h | https://commonvoice.mozilla.org/id |
| OpenSLR 35 (Large Javanese) | 185h | https://openslr.org/35/ |
| OpenSLR 36 (Large Sundanese) | 220h | https://openslr.org/36/ |
| OpenSLR 41 (High-quality ID) | 55h | https://openslr.org/41/ |
| OpenSLR 44 (Indonesian) | 170h | https://openslr.org/44/ |
| Google FLEURS (Indonesian) | ~12h | https://huggingface.co/datasets/google/fleurs |

> **Tip**: Combine multiple datasets. More speaker diversity = better voice cloning.

### Directory Structure

Place your raw data under `data/raw/`:

```
data/
├── raw/
│   ├── indonesian_train.tsv
│   ├── indonesian_valid.tsv    # (optional)
│   └── wavs/                   # Audio files referenced in TSVs
│       ├── speaker01/
│       ├── speaker02/
│       └── ...
├── manifests/                  # Generated by step 01
├── fbank/                      # Generated by step 02
└── tokens_id.txt               # Generated by step 03/04
```

## Tokenizer

We use **espeak-ng** with the Indonesian (`id`) locale for phonemization. This converts Indonesian text into IPA phoneme sequences that the model learns to map to speech.

Example:
```
"Selamat pagi"  →  ['s', 'ə', 'l', 'a', 'm', 'a', 't', ' ', 'p', 'a', 'ɡ', 'i']
```

### Building tokens.txt

The `tokens.txt` file maps each phoneme to an integer ID. You need to build this from your training data:

```bash
# Option A: Using the build_tokenizer tool (recommended)
python3 tools/build_tokenizer.py \
    --input data/raw/indonesian_train.tsv \
    --format tsv \
    --lang id \
    --output data/tokens_id.txt

# Option B: With English code-switching support
python3 tools/build_tokenizer.py \
    --input data/raw/indonesian_train.tsv \
    --format tsv \
    --lang id \
    --output data/tokens_id.txt \
    --add-english

# Option C: From an already-tokenized manifest (after step 03)
python3 scripts/04_generate_tokens_txt.py \
    --manifest data/fbank/indonesian_cuts_train_tokens.jsonl.gz \
    --output data/tokens_id.txt
```

The tokens.txt format is:
```
_	0
a	1
ə	2
i	3
...
```

Token `_` (pad) is always ID 0. Indonesian typically has ~60–80 unique phonemes; with English code-switching support, expect ~100–120.

## Training Pipeline

Run the scripts in order from the project root:

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Step 1: Convert TSV → Lhotse manifests
bash scripts/01_prepare_dataset.sh

# Step 2: Extract mel filterbank features (slow, CPU-heavy)
bash scripts/02_compute_fbank.sh

# Step 3: Phonemize text with espeak
bash scripts/03_prepare_tokens.sh

# Step 4: Generate tokens.txt (if not done via tools/build_tokenizer.py)
python3 scripts/04_generate_tokens_txt.py \
    --manifest data/fbank/indonesian_cuts_train_tokens.jsonl.gz \
    --output data/tokens_id.txt

# Step 5: Train the base model
bash scripts/05_train.sh

# Step 6: Distill to fewer steps (after base training converges)
bash scripts/06_train_distill.sh

# Step 7: Run inference
bash scripts/07_inference.sh
```

Edit the variables at the top of each script to match your setup (paths, dataset name, etc.).

### Monitoring Training

```bash
# TensorBoard
tensorboard --logdir exp/zipvoice-id --port 6006

# GPU monitoring on Jetson
sudo tegrastats

# Check latest checkpoint
ls -lt exp/zipvoice-id/*.pt | head -5
```

## Training on Jetson AGX Orin

The Jetson AGX Orin (64GB) is well-suited for training ZipVoice-ID:

### Memory Estimates

| Component | VRAM Usage |
|-----------|-----------|
| Model (FP16) | ~250MB |
| Optimizer states | ~1GB |
| Activations (max-duration=200) | ~3-5GB |
| **Total** | **~5-7GB** |

With 64GB unified memory, you have plenty of headroom. You can increase `max-duration` to 300-400 for faster training.

### Recommended Settings

```bash
# In scripts/05_train.sh, adjust:
MAX_DURATION=300    # Can go higher on 64GB Orin
NUM_EPOCHS=20       # Or more for larger datasets
```

### Performance Tips

1. **Use FP16** — already enabled (`--use-fp16 1`), halves memory
2. **Increase batch size** — raise `max-duration` until you see memory pressure
3. **Monitor thermals** — Jetson throttles at high temps
   ```bash
   sudo cat /sys/devices/virtual/thermal/thermal_zone*/temp
   ```
4. **Use NVMe storage** — fbank features are I/O heavy; avoid SD card
5. **Set power mode** to MAXN:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

### Expected Training Speed

With a 100h dataset on Jetson AGX Orin:
- ~2-4 epochs per day (depends on `max-duration`)
- Full training (20 epochs): ~5-10 days
- Distillation: ~2-3 additional days

## Distillation

Distillation trains a student model that matches the base (teacher) model's output in fewer ODE solver steps. This is critical for real-time inference.

### How It Works

1. Base model generates high-quality mel spectrograms in 50+ solver steps
2. Distillation trains the model to produce equivalent output in 4–8 steps
3. Quality is nearly identical; speed is 6–12x faster

### Running Distillation

```bash
# Edit scripts/06_train_distill.sh:
# - Set BASE_CHECKPOINT to your best base model checkpoint
# - Adjust NUM_EPOCHS (usually 10 is enough)

bash scripts/06_train_distill.sh
```

### Quality vs Speed Tradeoff

| Steps | Speed | Quality |
|-------|-------|---------|
| 4 | Fastest (~50ms/utterance) | Good, occasional artifacts |
| 8 | Fast (~100ms/utterance) | Very good |
| 16 | Moderate (~200ms/utterance) | Excellent |
| 50 | Slow (~500ms+/utterance) | Reference quality (base model) |

## Inference

### Basic Inference (24kHz)

```bash
# Edit scripts/07_inference.sh with your text and reference audio
bash scripts/07_inference.sh
```

Or use Python directly:

```python
from zipvoice.inference import ZipVoiceInference

model = ZipVoiceInference(
    checkpoint="exp/zipvoice-id-distill/best-valid-loss.pt",
    model_config="conf/zipvoice_base.json",
    token_file="data/tokens_id.txt",
    tokenizer="espeak",
    lang="id",
)

# Generate speech
wav = model.synthesize(
    text="Selamat datang di Indonesia.",
    reference_audio="data/reference/speaker.wav",
    num_steps=8,
)
# wav is a numpy array at 24kHz
```

## LuxTTS-Style Inference (48kHz)

To get higher quality 48kHz output like LuxTTS, use the **anchor sampler** and **LinaCodec vocoder**:

### What's Different

- **Anchor sampler**: Improved ODE solver that LuxTTS uses for better distilled model inference
- **LinaCodec**: Neural vocoder that upsamples 24kHz mel → 48kHz waveform with higher fidelity than standard Vocos

### Setup

```bash
# Install LinaCodec (if available separately)
pip install linacodec
# Or download from HuggingFace
```

### Python Example

```python
import torch
import torchaudio
from zipvoice.inference import ZipVoiceInference

# Load your trained model
model = ZipVoiceInference(
    checkpoint="exp/zipvoice-id-distill/best-valid-loss.pt",
    model_config="conf/zipvoice_base.json",
    token_file="data/tokens_id.txt",
    tokenizer="espeak",
    lang="id",
)

# Step 1: Generate mel features (not decoded audio)
mel = model.synthesize_mel(
    text="Halo, saya adalah sistem TTS bahasa Indonesia.",
    reference_audio="data/reference/speaker.wav",
    num_steps=8,
    # Use anchor sampler for better distilled inference:
    sampler="anchor",
)

# Step 2: Decode mel → 48kHz audio with LinaCodec
from linacodec import LinaCodecDecoder
vocoder = LinaCodecDecoder.from_pretrained("linacodec-48khz")
wav_48k = vocoder.decode(mel)  # Output: 48kHz waveform

# Save
torchaudio.save("output/generated_48khz.wav", wav_48k.unsqueeze(0), 48000)
```

> **Note**: The exact API for `synthesize_mel`, anchor sampler, and LinaCodec may vary depending on the ZipVoice and LuxTTS versions. Check their respective documentation for the current interface.

## Estimated Timeline

For a 100-hour Indonesian dataset on Jetson AGX Orin (64GB):

| Step | Time | Notes |
|------|------|-------|
| 01: Prepare dataset | 5–15 min | Fast, mostly I/O |
| 02: Compute fbank | 2–6 hours | CPU-bound, set `num-jobs` to nproc |
| 03: Prepare tokens | 30–60 min | Espeak phonemization |
| 04: Generate tokens.txt | 1–5 min | Quick scan |
| 05: Train base model | 5–10 days | ~2-4 epochs/day on Orin |
| 06: Distillation | 2–3 days | Fewer epochs needed |
| 07: Inference | Seconds | Per utterance |
| **Total** | **~1-2 weeks** | End-to-end |

For a 10-hour dataset, expect ~2-3 days total.

## Adapting to Other Languages

This recipe is designed for Indonesian but is easily adaptable:

1. **Change espeak locale**: Replace `id` with your target language code
   - Malay: `ms`
   - Javanese: `jv`
   - Sundanese: `su`
   - Thai: `th`
   - Vietnamese: `vi`

2. **Rebuild tokens.txt**: Run `tools/build_tokenizer.py` with `--lang <code>`

3. **Update script variables**: Change `LANG` and `DATASET` in all scripts

4. **Prepare your dataset**: Same TSV format, different language content

The model architecture and training process remain identical — only the phonemizer and data change.

---

## License

This training recipe is provided as-is. The ZipVoice model code has its own license — check the [ZipVoice repository](https://github.com/nickolay-kondratyev/ZipVoice).

## Acknowledgments

- [ZipVoice](https://github.com/nickolay-kondratyev/ZipVoice) — the model architecture
- [LuxTTS](https://huggingface.co/spaces/luxtts) — demonstrating ZipVoice at production quality
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — phonemization
- [Lhotse](https://github.com/lhotse-speech/lhotse) — data preparation
