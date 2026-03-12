#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Ground Truth and hypothesis txt files in two directories.
Computes Character Error Rate (CER) and Word Error Rate (WER) for each file
and writes results to a CSV file.
"""

import argparse
import csv
import os
import sys

import Levenshtein
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute CER and WER between Ground Truth and hypothesis txt files."
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Directory containing Ground Truth txt files.",
    )
    parser.add_argument(
        "--hyp",
        required=True,
        help="Directory containing hypothesis txt files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used when reading files (default: utf-8).",
    )
    return parser.parse_args()


def read_text(path, encoding):
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        print(f"Warning: could not read '{path}': {e}", file=sys.stderr)
        return None


def normalize_whitespace(text):
    """Replace all whitespace sequences (spaces, tabs, newlines, etc.) with a single space and strip."""
    return " ".join(text.split())


def compute_cer(gt_text, hyp_text):
    """Character Error Rate = edit_distance(gt, hyp) / len(gt)."""
    gt_chars = normalize_whitespace(gt_text)
    hyp_chars = normalize_whitespace(hyp_text)
    n = len(gt_chars)
    if n == 0:
        return 0.0, 0
    dist = Levenshtein.distance(gt_chars, hyp_chars)
    return dist / n, n


def compute_wer(gt_text, hyp_text):
    """Word Error Rate = word-level edit_distance(gt, hyp) / number_of_gt_words."""
    gt_words = normalize_whitespace(gt_text).split()
    hyp_words = normalize_whitespace(hyp_text).split()
    n = len(gt_words)
    if n == 0:
        return 0.0, 0
    # Use Levenshtein distance on the list of words via opcodes
    dist = _word_edit_distance(gt_words, hyp_words)
    return dist / n, n


def _word_edit_distance(seq1, seq2):
    """Compute edit distance between two sequences of words using dynamic programming."""
    len1, len2 = len(seq1), len(seq2)
    # Use two-row DP for memory efficiency
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[len2]


def main():
    args = parse_arguments()

    gt_files = {
        f for f in os.listdir(args.gt) if f.lower().endswith(".txt")
    }
    hyp_files = {
        f for f in os.listdir(args.hyp) if f.lower().endswith(".txt")
    }

    all_files = sorted(gt_files | hyp_files)

    if not all_files:
        print("No txt files found in the provided directories.", file=sys.stderr)
        sys.exit(1)

    rows = []

    for filename in tqdm(all_files, desc="Comparing files", unit="file"):
        gt_path = os.path.join(args.gt, filename)
        hyp_path = os.path.join(args.hyp, filename)

        if not os.path.exists(gt_path):
            print(f"Warning: '{filename}' missing in GT directory, skipping.", file=sys.stderr)
            continue
        if not os.path.exists(hyp_path):
            print(f"Warning: '{filename}' missing in hypothesis directory, skipping.", file=sys.stderr)
            continue

        gt_text = read_text(gt_path, args.encoding)
        hyp_text = read_text(hyp_path, args.encoding)

        if gt_text is None or hyp_text is None:
            continue

        cer, n_chars = compute_cer(gt_text, hyp_text)
        wer, n_words = compute_wer(gt_text, hyp_text)

        rows.append(
            {
                "file_name": filename,
                "cer": f"{cer:.6f}",
                "wer": f"{wer:.6f}",
                "number_of_gt_characters": n_chars,
                "number_of_gt_words": n_words,
            }
        )

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "file_name",
            "cer",
            "wer",
            "number_of_gt_characters",
            "number_of_gt_words",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to '{args.output}' ({len(rows)} files processed).")

    # Print overall summary
    total_chars = sum(int(r["number_of_gt_characters"]) for r in rows)
    total_words = sum(int(r["number_of_gt_words"]) for r in rows)
    if total_chars > 0:
        avg_cer = sum(float(r["cer"]) * int(r["number_of_gt_characters"]) for r in rows) / total_chars
        print(f"Overall CER (weighted): {avg_cer * 100:.2f} %")
    if total_words > 0:
        avg_wer = sum(float(r["wer"]) * int(r["number_of_gt_words"]) for r in rows) / total_words
        print(f"Overall WER (weighted): {avg_wer * 100:.2f} %")


if __name__ == "__main__":
    main()

