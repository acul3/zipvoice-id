#!/usr/bin/env python3
"""
build_tokenizer.py — Build tokens.txt for Indonesian (espeak-based)
===================================================================

Runs espeak-ng phonemization over a corpus of Indonesian text and produces
a tokens.txt file compatible with ZipVoice's EspeakTokenizer.

Usage:
    # From a text file (one sentence per line):
    python3 tools/build_tokenizer.py \
        --input data/raw/texts.txt \
        --lang id \
        --output data/tokens_id.txt

    # From a TSV file (id<TAB>text<TAB>wav_path):
    python3 tools/build_tokenizer.py \
        --input data/raw/indonesian_train.tsv \
        --format tsv \
        --lang id \
        --output data/tokens_id.txt

    # With English phonemes for code-switching:
    python3 tools/build_tokenizer.py \
        --input data/raw/texts.txt \
        --lang id \
        --add-english \
        --output data/tokens_id.txt

Output format (tokens.txt):
    _       0
    a       1
    ə       2
    ...
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from piper_phonemize import phonemize_espeak
except ImportError:
    print(
        "ERROR: piper_phonemize not installed.\n"
        "Install with: pip install piper_phonemize -f "
        "https://k2-fsa.github.io/icefall/piper_phonemize.html",
        file=sys.stderr,
    )
    sys.exit(1)


def read_texts(input_path: str, fmt: str = "txt") -> list:
    """Read texts from a file."""
    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if fmt == "tsv":
                parts = line.split("\t")
                if len(parts) >= 2:
                    texts.append(parts[1])  # text is 2nd column
            else:
                texts.append(line)
    return texts


def phonemize_texts(texts: list, lang: str = "id") -> Counter:
    """Run espeak phonemization and count all tokens."""
    counter = Counter()
    errors = 0

    for i, text in enumerate(texts):
        try:
            phoneme_lists = phonemize_espeak(text, lang)
            for phone_list in phoneme_lists:
                for phone in phone_list:
                    if phone.strip():
                        counter[phone] += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning line {i}: {e}", file=sys.stderr)
            elif errors == 6:
                print("  (suppressing further warnings)", file=sys.stderr)

    if errors:
        print(f"\n  Total phonemization errors: {errors}/{len(texts)}", file=sys.stderr)

    return counter


def add_english_phonemes(counter: Counter, lang: str = "en-us"):
    """Run espeak on common English sentences to capture English phonemes."""
    english_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells sea shells by the sea shore.",
        "How much wood would a woodchuck chuck?",
        "Peter Piper picked a peck of pickled peppers.",
        "Unique New York, you know you need unique New York.",
        "Red lorry, yellow lorry, red lorry, yellow lorry.",
        "The sixth sick sheikh's sixth sheep's sick.",
        "I scream, you scream, we all scream for ice cream.",
        "Technology, mathematics, philosophy, psychology.",
        "Beautiful, wonderful, comfortable, vegetable.",
    ]
    for text in english_samples:
        try:
            phoneme_lists = phonemize_espeak(text, lang)
            for phone_list in phoneme_lists:
                for phone in phone_list:
                    if phone.strip() and phone not in counter:
                        counter[phone] = 1  # just register existence
        except Exception:
            pass


def write_tokens(counter: Counter, output_path: str):
    """Write tokens.txt file."""
    # Special tokens first
    special = ["_"]  # pad = id 0

    # Sort phonemes by frequency (most common first)
    phonemes = [tok for tok, _ in counter.most_common() if tok and tok not in special]

    all_tokens = special + phonemes

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, token in enumerate(all_tokens):
            f.write(f"{token}\t{idx}\n")

    return all_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Build tokens.txt for ZipVoice Indonesian TTS"
    )
    parser.add_argument("--input", required=True, help="Input text/TSV file")
    parser.add_argument(
        "--format",
        choices=["txt", "tsv"],
        default="txt",
        help="Input format: txt (one sentence per line) or tsv",
    )
    parser.add_argument("--lang", default="id", help="Espeak language code")
    parser.add_argument("--output", default="data/tokens_id.txt", help="Output path")
    parser.add_argument(
        "--add-english",
        action="store_true",
        help="Include English phonemes for code-switching",
    )
    args = parser.parse_args()

    print(f"Reading texts from: {args.input} (format={args.format})")
    texts = read_texts(args.input, args.format)
    print(f"  Found {len(texts)} texts")

    print(f"\nPhonemizing with espeak lang={args.lang}...")
    counter = phonemize_texts(texts, args.lang)
    print(f"  Found {len(counter)} unique phoneme tokens")
    print(f"  Total token occurrences: {sum(counter.values()):,}")

    if args.add_english:
        before = len(counter)
        add_english_phonemes(counter)
        print(f"\n  Added {len(counter) - before} English-only phonemes")

    all_tokens = write_tokens(counter, args.output)

    print(f"\n{'='*50}")
    print(f"tokens.txt → {args.output}")
    print(f"Vocabulary size: {len(all_tokens)}")
    print(f"  Special: 1 (pad '_')")
    print(f"  Phonemes: {len(all_tokens) - 1}")
    print(f"\nTop 30 phonemes:")
    for tok, count in counter.most_common(30):
        print(f"  {tok!r:12s}  {count:>8,}")


if __name__ == "__main__":
    main()
