#!/usr/bin/env python3
"""
Download a HuggingFace model using wget (resumable, fast).

Uses huggingface_hub to list repo files, then wget -c for all downloads.
This bypasses the XetHub CAS storage layer which is significantly slower
on many networks. See: https://github.com/huggingface/xet-core/issues/800

Usage:
  python3 download_model.py
  python3 download_model.py Qwen/Qwen2.5-7B-Instruct
  python3 download_model.py meta-llama/Llama-3.1-8B-Instruct --dir ~/.models
  python3 download_model.py Qwen/Qwen2.5-72B --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from huggingface_hub import list_repo_tree
from huggingface_hub.hf_api import RepoFile

HF_BASE = "https://huggingface.co"


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download(url: str, dest: Path):
    print(f"  {dest.name}")
    result = subprocess.run(["wget", "-c", "--show-progress", "-q", url, "-O", str(dest)])
    if result.returncode != 0:
        print(f"ERROR: wget failed for {url}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--dir", default=None,
                        help="Output directory (default: .models/<model-name>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files and total size without downloading")
    args = parser.parse_args()

    model_id = args.model
    model_name = model_id.split("/")[-1].lower()
    out_dir = Path(args.dir) if args.dir else Path(".models") / model_name

    print(f"Fetching file list for {model_id} ...")
    try:
        entries = [e for e in list_repo_tree(model_id, recursive=True) if isinstance(e, RepoFile) and "/" not in e.rfilename]
    except Exception as e:
        print(f"ERROR: could not fetch file list: {e}", file=sys.stderr)
        sys.exit(1)

    shards = [e for e in entries if e.rfilename.endswith(".safetensors") or e.rfilename.endswith(".bin")]
    config = [e for e in entries if e not in shards]

    if args.dry_run:
        print(f"\n{'File':<50} {'Size':>10}")
        print("-" * 62)
        total = 0
        for e in config + shards:
            size = e.size or 0
            total += size
            print(f"  {e.rfilename:<48} {fmt_size(size):>10}")
        print("-" * 62)
        print(f"  {'TOTAL':<48} {fmt_size(total):>10}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"{HF_BASE}/{model_id}/resolve/main"

    print(f"\nDownloading config/tokenizer files to {out_dir} ...")
    for e in config:
        download(f"{base_url}/{e.rfilename}", out_dir / e.rfilename)

    print(f"\nDownloading model shards ({len(shards)} files) ...")
    for e in shards:
        download(f"{base_url}/{e.rfilename}", out_dir / e.rfilename)

    print(f"\nDone. Model saved to: {out_dir.resolve()}")
    print(f"\nRun with:")
    print(f"  python3 chat.py --model {out_dir.resolve()}")


if __name__ == "__main__":
    main()
