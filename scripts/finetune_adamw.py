"""Fine-tune pretrained ZipVoice with AdamW + torch.amp mixed precision."""
import json, logging, os, sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from lhotse import load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.dataset.dataset import SpeechSynthesisDataset
from zipvoice.dataset.datamodule import _SeedWorkers
from zipvoice.utils.checkpoint import load_checkpoint
from lhotse.dataset.input_strategies import PrecomputedFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Config
EXP_DIR = Path("exp/zipvoice-id-finetune")
EXP_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = "pretrained/zipvoice/model.pt"
MODEL_CONFIG = "pretrained/zipvoice/model.json"
TOKEN_FILE = "pretrained/zipvoice/tokens.txt"
TRAIN_MANIFEST = "data/fbank_audiobook/audiobook_cuts_train_tokens.jsonl.gz"
DEV_MANIFEST = "data/fbank_audiobook/audiobook_cuts_valid_tokens.jsonl.gz"

LR = 1e-5
NUM_EPOCHS = 5
MAX_DURATION = 200
FEAT_SCALE = 0.1
SAVE_EVERY = 500
LOG_EVERY = 50

device = torch.device("cuda")
fix_random_seed(42)

# Load model
with open(MODEL_CONFIG) as f:
    model_config = json.load(f)

tokenizer = EspeakTokenizer(token_file=TOKEN_FILE, lang="id")
tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

model = ZipVoice(**model_config["model"], **tokenizer_config)
load_checkpoint(filename=CHECKPOINT, model=model, strict=True)
model = model.to(device)
logging.info(f"Loaded model: {sum(p.numel() for p in model.parameters())} params")

# AdamW optimizer in fp32 (master weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# PyTorch native AMP scaler (NOT ScaledAdam)
scaler = GradScaler("cuda")

# Data
def tokenize_text(c, tokenizer):
    if hasattr(c.supervisions[0], "tokens"):
        tokens = tokenizer.tokens_to_token_ids([c.supervisions[0].tokens])
    else:
        tokens = tokenizer.texts_to_token_ids([c.supervisions[0].text])
    c.supervisions[0].tokens = tokens[0]
    return c

train_cuts = load_manifest_lazy(TRAIN_MANIFEST)
dev_cuts = load_manifest_lazy(DEV_MANIFEST)

_tok = partial(tokenize_text, tokenizer=tokenizer)
train_cuts = train_cuts.map(_tok)
dev_cuts = dev_cuts.map(_tok)

train_ds = SpeechSynthesisDataset(return_text=True, return_tokens=True, return_spk_ids=True, feature_input_strategy=PrecomputedFeatures())
dev_ds = SpeechSynthesisDataset(return_text=True, return_tokens=True, return_spk_ids=True, feature_input_strategy=PrecomputedFeatures())

train_sampler = DynamicBucketingSampler(train_cuts, max_duration=MAX_DURATION, shuffle=True, num_buckets=30)
dev_sampler = DynamicBucketingSampler(dev_cuts, max_duration=MAX_DURATION, shuffle=False)

train_dl = DataLoader(train_ds, sampler=train_sampler, batch_size=None, num_workers=4, persistent_workers=False, worker_init_fn=_SeedWorkers(42))
dev_dl = DataLoader(dev_ds, sampler=dev_sampler, batch_size=None, num_workers=2, persistent_workers=False)

tb_writer = SummaryWriter(log_dir=f"{EXP_DIR}/tensorboard")

# Training
global_step = 0
best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    fix_random_seed(42 + epoch)
    train_dl.sampler.set_epoch(epoch - 1)
    
    tot_loss = 0
    tot_frames = 0
    
    for batch_idx, batch in enumerate(train_dl):
        features = batch["features"].to(device) * FEAT_SCALE
        features_lens = batch["features_lens"].to(device)
        tokens = batch["tokens"]
        
        B, T, F = features.shape
        noise = torch.randn_like(features)
        t = torch.rand(B, 1, 1, device=device)
        
        optimizer.zero_grad()
        
        with autocast("cuda", dtype=torch.float16):
            loss = model(tokens=tokens, features=features, features_lens=features_lens, noise=noise, t=t, condition_drop_ratio=0.2)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        frames = features_lens.sum().item()
        tot_loss += loss.item() * frames
        tot_frames += frames
        global_step += 1
        
        if global_step % LOG_EVERY == 0:
            avg = tot_loss / max(tot_frames, 1)
            mem = torch.cuda.max_memory_allocated() // 1024**2
            logging.info(f"Epoch {epoch}, step {global_step}, batch {batch_idx}, loss={loss.item():.4f}, avg={avg:.4f}, lr={LR}, mem={mem}MB, grad_scale={scaler.get_scale():.0f}")
            tb_writer.add_scalar("train/loss", loss.item(), global_step)
        
        if global_step % SAVE_EVERY == 0:
            ckpt = EXP_DIR / f"checkpoint-{global_step}.pt"
            torch.save(model.state_dict(), ckpt)
            logging.info(f"Saved {ckpt}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_frames = 0
    with torch.no_grad():
        for batch in dev_dl:
            features = batch["features"].to(device) * FEAT_SCALE
            features_lens = batch["features_lens"].to(device)
            tokens = batch["tokens"]
            B, T, F = features.shape
            noise = torch.randn_like(features)
            t = (torch.arange(B, device=device).float() / B).unsqueeze(1).unsqueeze(2)
            with autocast("cuda", dtype=torch.float16):
                loss = model(tokens=tokens, features=features, features_lens=features_lens, noise=noise, t=t, condition_drop_ratio=0.0)
            frames = features_lens.sum().item()
            val_loss += loss.item() * frames
            val_frames += frames
    
    val_avg = val_loss / max(val_frames, 1)
    logging.info(f"Epoch {epoch} validation: loss={val_avg:.4f}")
    tb_writer.add_scalar("val/loss", val_avg, global_step)
    
    ckpt = EXP_DIR / f"epoch-{epoch}.pt"
    torch.save(model.state_dict(), ckpt)
    if val_avg < best_val_loss:
        best_val_loss = val_avg
        torch.save(model.state_dict(), EXP_DIR / "best-valid-loss.pt")
        logging.info(f"New best val loss: {val_avg:.4f}")

logging.info("Done!")
