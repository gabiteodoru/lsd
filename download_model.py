#!/usr/bin/env python3
"""
Download a HuggingFace model using wget (resumable, fast).

Uses huggingface_hub to list repo files, then wget -c for all downloads.
This bypasses the XetHub CAS storage layer which is significantly slower
on many networks. See: https://github.com/huggingface/xet-core/issues/800

Usage:
  python3 download_model.py Qwen/Qwen2.5-7B-Instruct
  python3 download_model.py meta-llama/Llama-3.1-8B-Instruct --dir ~/.models
"""

import argparse
import subprocess
import sys
from pathlib import Path
from huggingface_hub import list_repo_files

HF_BASE = "https://huggingface.co"


def download(url: str, dest: Path):
    print(f"  {dest.name}")
    result = subprocess.run(["wget", "-c", "--show-progress", "-q", url, "-O", str(dest)])
    if result.returncode != 0:
        print(f"ERROR: wget failed for {url}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HuggingFace model ID, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dir", default=None,
                        help="Output directory (default: .models/<model-name>)")
    args = parser.parse_args()

    model_id = args.model
    model_name = model_id.split("/")[-1].lower()
    out_dir = Path(args.dir) if args.dir else Path(".models") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file list for {model_id} ...")
    try:
        files = list(list_repo_files(model_id))
    except Exception as e:
        print(f"ERROR: could not fetch file list: {e}", file=sys.stderr)
        sys.exit(1)

    # Split into shards (large) and everything else (config/tokenizer)
    shards = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin")]
    config = [f for f in files if f not in shards and "/" not in f]  # skip subdirectories

    base_url = f"{HF_BASE}/{model_id}/resolve/main"

    print(f"\nDownloading config/tokenizer files to {out_dir} ...")
    for f in config:
        download(f"{base_url}/{f}", out_dir / f)

    print(f"\nDownloading model shards ({len(shards)} files) ...")
    for f in shards:
        download(f"{base_url}/{f}", out_dir / f)

    print(f"\nDone. Model saved to: {out_dir.resolve()}")
    print(f"\nRun with:")
    print(f"  python3 chat.py --model {out_dir.resolve()}")


if __name__ == "__main__":
    main()
