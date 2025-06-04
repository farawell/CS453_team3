#!/usr/bin/env python3
"""
sbert_prompts.py

Compute the SBERT‐based semantic distance (cosine distance) between a reference prompt
and a set of other prompts, where each prompt is stored in its own `.txt` or `.json` file.

Usage:
    python sbert_prompts.py -r /path/to/A.txt -d /path/to/prompts_dir/
    python sbert_prompts.py -r /path/to/A.json -d /path/to/prompts_dir/

This will print the cosine distance (1 – cosine_similarity) between A.txt (or A.json)
and each other .txt/.json file in prompts_dir/.

TODO: Check the compatibility with promptpex

Author: Yohan Park
"""

import os
import argparse
import re
import json
from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer

def normalize_text(text: str) -> str:
    """
    Normalize a natural-language prompt for more consistent embedding:
    - Strip leading/trailing whitespace
    - Collapse all internal whitespace (spaces, newlines, tabs) to a single space
    - Convert to lowercase

    Args:
        text: Original prompt string.
    Returns:
        A normalized string.
    """
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_prompts_from_dir(directory: str) -> Dict[str, str]:
    """
    Read all .txt and .json files from a directory. Each file is assumed to contain
    exactly one prompt. Returns a mapping from filename → normalized prompt string.

    - For .txt: read the entire file as a raw prompt.
    - For .json: expect a top-level key "prompt" whose value is the prompt string.

    Args:
        directory: Path to directory containing .txt/.json prompt files.
    Returns:
        A dict where keys are filenames (basename) and values are normalized texts.
    """
    prompts: Dict[str, str] = {}

    for entry in os.listdir(directory):
        lower = entry.lower()
        path = os.path.join(directory, entry)
        if not os.path.isfile(path):
            continue

        try:
            if lower.endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    raw = f.read()

            elif lower.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract a prompt string from the JSON object
                if isinstance(data, dict) and 'prompt' in data and isinstance(data['prompt'], str):
                    raw = data['prompt']
                else:
                    raise ValueError(f"JSON file '{entry}' does not contain a top-level string under key 'prompt'.")

            else:
                # skip any other extensions
                continue

            normalized = normalize_text(raw)
            prompts[entry] = normalized

        except Exception as e:
            # If anything goes wrong (malformed JSON, missing 'prompt' key, etc.), skip
            print(f"Warning: skipping '{entry}': {e}")

    return prompts

def main():
    parser = argparse.ArgumentParser(
        description="Compute SBERT‐based cosine distance between a reference prompt and a set of other prompts."
    )
    parser.add_argument(
        '--reference', '-r',
        type=str,
        required=True,
        help="Path to the reference file (A.txt or A.json)."
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        required=True,
        help="Directory containing all prompt files (.txt and/or .json), including the reference."
    )
    args = parser.parse_args()

    ref_path = args.reference
    prompt_dir = args.dir

    # Validate reference file exists
    if not os.path.isfile(ref_path):
        parser.error(f"Reference file not found: {ref_path}")

    ref_name = os.path.basename(ref_path)

    # Load and normalize all prompts from the directory
    prompts = load_prompts_from_dir(prompt_dir)
    if ref_name not in prompts:
        parser.error(
            f"Reference file '{ref_name}' not found among .txt/.json files in '{prompt_dir}'."
        )

    # Separate reference prompt from the rest
    ref_text = prompts.pop(ref_name)
    other_prompts = prompts  # dict: filename → normalized text

    # Prepare filenames in sorted order for consistent batching
    filenames = sorted(other_prompts.keys())
    texts = [ref_text] + [other_prompts[f] for f in filenames]

    # Load SBERT model and compute embeddings (normalized)
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    ref_emb = embeddings[0]            # reference embedding
    other_embs = embeddings[1:]        # embeddings for the other prompts

    # Compute cosine distances: dist = 1 – (ref_emb · other_emb)
    results = []
    for fname, emb in zip(filenames, other_embs):
        dist = 1.0 - np.dot(ref_emb, emb)
        results.append((fname, dist))

    # Sort by ascending distance (most similar first)
    results.sort(key=lambda x: x[1])

    # Print results in the same table format as before
    print(f"Reference file: '{ref_name}'\n")
    print(f"{'Other File':<30}  {'Distance':>12}")
    print(f"{'-'*30}  {'-'*12}")
    for fname, dist in results:
        print(f"{fname:<30}  {dist:>12.4f}")

if __name__ == "__main__":
    main()