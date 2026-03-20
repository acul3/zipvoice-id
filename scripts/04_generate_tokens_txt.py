#!/usr/bin/env python3
"""
04_generate_tokens_txt.py
=========================
Scans a tokenized Lhotse manifest (or raw TSV) and collects all unique
espeak phoneme tokens for Indonesian, then writes tokens.txt.

This is an alternative to tools/build_tokenizer.py — it works directly
from an already-tokenized manifest (output of step 03) rather than
re-running espeak from scratch.

Usage:
    # From tokenized manifest (preferred, post step 03):
    python3 scripts/04_generate_tokens_txt.py \
        --manifest data/fbank/indonesian_cuts_train_tokens.jsonl.gz \
        --output data/tokens_id.txt

    # From raw TSV (runs espeak internally):
    python3 scripts/04_generate_tokens_txt.py \
        --tsv data/raw/indonesian_train.tsv \
        --lang id \
        --output data/tokens_id.txt \
        --add-english   # Add English phonemes for code-switching
"""

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path


def collect_from_manifest(manifest_path: str) -> Counter:
    """Read tokens from a tokenized lhotse manifest (.jsonl.gz)."""
    counter = Counter()
    path = Path(manifest_path)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            cut = json.loads(line)
            # Lhotse cuts store supervisions with custom_data or tokens field
            for sup in cut.get("supervisions", []):
                tokens = sup.get("tokens") or sup.get("custom", {}).get("tokens", [])
                if isinstance(tokens, list):
                    for tok in tokens:
                        if isinstance(tok, str):
                            counter[tok] += 1
                        elif isinstance(tok, dict):
                            counter[tok.get("symbol", "")] += 1

    return counter


def collect_from_tsv(tsv_path: str, lang: str = "id") -> Counter:
    """Phonemize texts from a TSV file using espeak and count tokens."""
    try:
        from piper_phonemize import phonemize_espeak
    except ImportError:
        print("ERROR: piper_phonemize not installed. Run: pip install piper_phonemize")
        sys.exit(1)

    counter = Counter()
    path = Path(tsv_path)

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            text = parts[1]
            try:
                phonemes = phonemize_espeak(text, lang)
                for phone_list in phonemes:
                    for phone in phone_list:
                        if phone.strip():
                            counter[phone] += 1
            except Exception as e:
                print(f"Warning: line {i}: {e}", file=sys.stderr)

    return counter


def get_english_phonemes() -> list:
    """Return a standard set of English IPA phonemes for code-switching support."""
    return [
        "p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "v", "θ", "ð",
        "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "r", "w", "j",
        "iː", "ɪ", "eɪ", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "oʊ", "ʊ", "uː",
        "ʌ", "ɜː", "ə", "aɪ", "aʊ", "ɔɪ",
    ]


def write_tokens(counter: Counter, output_path: str, add_english: bool = False):
    """Write tokens.txt with special tokens first, sorted by frequency."""
    special_tokens = ["_"]  # pad at id=0

    # Gather phoneme tokens, sorted by frequency descending
    phoneme_tokens = [tok for tok, _ in counter.most_common() if tok and tok != "_"]

    if add_english:
        eng = get_english_phonemes()
        for tok in eng:
            if tok not in phoneme_tokens:
                phoneme_tokens.append(tok)

    all_tokens = special_tokens + phoneme_tokens

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, token in enumerate(all_tokens):
            f.write(f"{token}\t{idx}\n")

    print(f"\n=== tokens.txt written to {output_path} ===")
    print(f"Vocabulary size: {len(all_tokens)}")
    print(f"  Special tokens: {len(special_tokens)}")
    print(f"  Phoneme tokens: {len(all_tokens) - len(special_tokens)}")
    print("\nTop 20 most common phonemes:")
    for tok, count in counter.most_common(20):
        print(f"  {tok!r:12s}  {count:8d}")


def main():
    parser = argparse.ArgumentParser(description="Generate tokens.txt for ZipVoice-ID")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="Tokenized lhotse manifest (.jsonl.gz)")
    group.add_argument("--tsv", help="Raw TSV file (will run espeak internally)")

    parser.add_argument("--lang", default="id", help="Espeak language code (default: id)")
    parser.add_argument("--output", default="data/tokens_id.txt", help="Output tokens.txt path")
    parser.add_argument("--add-english", action="store_true",
                        help="Add English phonemes for code-switching support")
    args = parser.parse_args()

    if args.manifest:
        print(f"Collecting tokens from manifest: {args.manifest}")
        counter = collect_from_manifest(args.manifest)
    else:
        print(f"Phonemizing TSV with espeak lang={args.lang}: {args.tsv}")
        counter = collect_from_tsv(args.tsv, args.lang)

    if not counter:
        print("ERROR: No tokens found! Check your input file format.", file=sys.stderr)
        sys.exit(1)

    write_tokens(counter, args.output, add_english=args.add_english)


if __name__ == "__main__":
    main()
