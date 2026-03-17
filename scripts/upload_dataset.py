#!/usr/bin/env python3
# Requirements: huggingface_hub
# Install: pip install huggingface_hub

import argparse
import json
import sys
import tempfile
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Upload EOT research output to Hugging Face Hub")
    parser.add_argument("--input", default="research_output.json", help="Path to research JSON file")
    parser.add_argument("--repo", required=True, help="HuggingFace repo id (e.g. username/dataset-name)")
    parser.add_argument("--token", default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    # Read and parse input
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    records = [
        {
            "run_index": r.get("run_index"),
            "token_count": r.get("token_count"),
            "transformed_count": r.get("transformed_count"),
            "avg_confidence": r.get("avg_confidence"),
            "avg_perplexity": r.get("avg_perplexity"),
            "vocab_diversity": r.get("vocab_diversity"),
            "prompt": data.get("prompt", ""),
            "provider": data.get("provider", ""),
            "transform": data.get("transform", ""),
        }
        for r in runs
    ]

    if args.dry_run:
        print(f"[dry-run] Would upload {len(records)} records to repo: {args.repo}")
        print(f"[dry-run] Source file: {args.input}")
        print(f"[dry-run] Sample record: {json.dumps(records[0], indent=2) if records else 'none'}")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    # Write records as JSONL to a temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for rec in records:
            tmp.write(json.dumps(rec) + "\n")
        tmp_path = tmp.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="data/runs.jsonl",
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"Uploaded {len(records)} records to {args.repo}/data/runs.jsonl")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
