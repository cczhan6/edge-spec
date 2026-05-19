#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/"
    "refs/heads/main/data/spec_bench/question.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SpecBench question.jsonl.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="data/spec_bench/question.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(args.url, output)
    print(f"downloaded {args.url} -> {output}")


if __name__ == "__main__":
    main()
