#!/usr/bin/env python3
"""
Download a HuggingFace model with parallel resumable downloads.

Usage:
  python3 download_model.py
  python3 download_model.py Qwen/Qwen2.5-7B-Instruct
  python3 download_model.py meta-llama/Llama-3.1-8B-Instruct --dir ~/.models
  python3 download_model.py Qwen/Qwen2.5-72B --dry-run
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import list_repo_tree
from huggingface_hub.hf_api import RepoFile
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

HF_BASE = "https://huggingface.co"
CHUNK = 1024 * 1024  # 1 MB


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download(url: str, dest: Path, size: int, progress: Progress):
    resume = dest.stat().st_size if dest.exists() else 0
    if resume >= size:
        return

    task_id = progress.add_task("download", name=dest.name, total=size, completed=resume)
    headers = {"Range": f"bytes={resume}-"} if resume else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("ab" if resume else "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                progress.update(task_id, advance=len(chunk))
    progress.remove_task(task_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--dir", default=None,
                        help="Output directory (default: .models/<model-name>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files and total size without downloading")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel shard downloads (default: 4)")
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

    progress = Progress(
        TextColumn("[bold]{task.fields[name]}", justify="right"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        print(f"Downloading config/tokenizer files to {out_dir} ...")
        for e in config:
            download(f"{base_url}/{e.rfilename}", out_dir / e.rfilename, e.size or 0, progress)

        print(f"\nDownloading model shards ({len(shards)} files, {args.workers} parallel) ...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download, f"{base_url}/{e.rfilename}", out_dir / e.rfilename, e.size or 0, progress): e
                for e in shards
            }
            for fut in as_completed(futures):
                fut.result()

    print(f"\nDone. Model saved to: {out_dir.resolve()}")
    print(f"\nRun with:")
    print(f"  python3 chat.py --model {out_dir.resolve()}")


if __name__ == "__main__":
    main()
