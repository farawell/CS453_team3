#!/usr/bin/env python3
"""
ncd_prompts.py

Compute the Normalized Compression Distance (NCD) between a reference prompt
and a set of other prompts, where each prompt is stored in its own `.txt` or `.json` file.

Usage:
    python ncd_prompts.py -r /path/to/A.txt -d /path/to/prompts_dir/
    python ncd_prompts.py -r /path/to/A.json -d /path/to/prompts_dir/

This will print the NCD between A.txt (or A.json) and each other .txt/.json file
in prompts_dir/, sorted by ascending distance (most similar first).

TODO: Check the compatibility with promptpex

Author: Yohan Park
"""

import os
import argparse
import zlib
import re
import json
from typing import Dict

def normalize_text(text: str) -> str:
    """
    Normalize a natural-language prompt for more consistent compression:
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

def compressed_size(s: str, level: int = 9) -> int:
    """
    Compute the compressed size (in bytes) of a UTF-8 string using zlib.
    We use zlib at the highest compression level for best approximation
    of shared patterns.

    Args:
        s: The input string to compress.
        level: Compression level (0–9); 9 is slowest but highest compression.
    Returns:
        The length (in bytes) of the zlib-compressed data.
    """
    b = s.encode('utf-8')
    c = zlib.compress(b, level=level)
    return len(c)

def ncd(x: str, y: str, cache: Dict[str, int] = None) -> float:
    """
    Calculate the Normalized Compression Distance between two texts x and y:
        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    where C(z) is the compressed size of z.

    We optionally accept a cache dict mapping text→compressed_size to avoid
    recomputing C(x) for the same string multiple times.

    Args:
        x: First normalized text.
        y: Second normalized text.
        cache: Optional dict mapping strings → their compressed size.
    Returns:
        A float in [0, 1] (or slightly above 1 due to compressor overhead).
    """
    if cache is None:
        cache = {}

    # Compute or retrieve C(x)
    if x in cache:
        cx = cache[x]
    else:
        cx = compressed_size(x)
        cache[x] = cx

    # Compute or retrieve C(y)
    if y in cache:
        cy = cache[y]
    else:
        cy = compressed_size(y)
        cache[y] = cy

    # If either string is empty (cx or cy == 0), return maximal distance
    if cx == 0 or cy == 0:
        return 1.0

    # Concatenate with a newline delimiter to avoid accidental merging
    xy = x + "\n" + y
    if xy in cache:
        cxy = cache[xy]
    else:
        cxy = compressed_size(xy)
        cache[xy] = cxy

    ncd_value = (cxy - min(cx, cy)) / max(cx, cy)
    return max(0.0, min(1.0, ncd_value))

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
        description="Compute NCD between a reference prompt and a set of other prompts."
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

    # Precompute C(A) once by caching
    cache: Dict[str, int] = {}
    cache[ref_text] = compressed_size(ref_text, level=9)

    # Compute NCD(A, B_i) for each B_i
    results = []
    for fname, b_text in other_prompts.items():
        if b_text not in cache:
            cache[b_text] = compressed_size(b_text, level=9)
        dist = ncd(ref_text, b_text, cache=cache)
        results.append((fname, dist))

    # Sort by ascending distance (most similar first)
    results.sort(key=lambda x: x[1])

    # Print results in ascending order
    print(f"Reference file: '{ref_name}'\n")
    print(f"{'Other File':<30}  {'NCD Distance':>12}")
    print(f"{'-'*30}  {'-'*12}")
    for fname, dist in results:
        print(f"{fname:<30}  {dist:>12.4f}")

if __name__ == "__main__":
    main()