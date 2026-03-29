#!/usr/bin/env python3
"""Verify compaction quality by showing unified diffs of labeled records.

Reads the last N records from output.jsonl, reconstructs before/after,
prints unified diff + stats, optionally opens in Cursor diff view.

Usage:
    # Show diff of last labeled record
    python3 train/labeling/verify_compaction.py

    # Show last 5 records
    python3 train/labeling/verify_compaction.py --last 5

    # Open in Cursor diff view
    python3 train/labeling/verify_compaction.py --cursor

    # Check a specific record ID
    python3 train/labeling/verify_compaction.py --id 30042
"""

import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "labeling_tasks")
OUTPUT = os.path.join(BASE, "output.jsonl")

CURSOR_CLI = "/home/tejas/.cursor-server/bin/linux-x64/b29eb4ee5f9f6d1cb2afbc09070198d3ea6ad760/bin/remote-cli/cursor"


def expand_kept_frags(kept_frags: list) -> set:
    """Expand kept_frags (ints and [start, end] ranges) into a set of line numbers."""
    lines = set()
    for item in kept_frags:
        if isinstance(item, int):
            lines.add(item)
        elif isinstance(item, list) and len(item) == 2:
            start, end = int(item[0]), int(item[1])
            lines.update(range(start, end + 1))
    return lines


def reconstruct_after(code: str, kept_frags: list) -> str:
    """Reconstruct the compacted version from code + kept_frags."""
    all_lines = code.split("\n")
    kept_set = expand_kept_frags(kept_frags)
    kept_lines = []
    for i, line in enumerate(all_lines, 1):
        if i in kept_set:
            kept_lines.append(line)
    return "\n".join(kept_lines)


def load_records(output_path: str, last_n: int = None, record_id: str = None) -> list:
    """Load records from output.jsonl."""
    records = []
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if record_id and str(rec.get("id")) != str(record_id):
                    continue
                records.append(rec)
            except json.JSONDecodeError:
                continue
    if last_n and not record_id:
        records = records[-last_n:]
    return records


def show_diff(rec: dict, open_cursor: bool = False):
    """Show unified diff for a single record."""
    code = rec.get("code", "")
    kept_frags = rec.get("kept_frags", [])
    query = rec.get("query", "")
    rec_id = rec.get("id", "?")
    source = rec.get("source", "?")

    all_lines = code.split("\n")
    total = len(all_lines)
    kept_set = expand_kept_frags(kept_frags)
    kept_count = len(kept_set)
    ratio = kept_count / total if total > 0 else 0

    before_lines = all_lines
    after_lines = [line for i, line in enumerate(all_lines, 1) if i in kept_set]

    print(f"\n{'='*70}")
    print(f"ID: {rec_id} | Source: {source}")
    print(f"Query: {query[:100]}")
    print(f"Lines: {total} -> {kept_count} ({ratio:.0%} kept, {100*(1-ratio):.0%} dropped)")
    print(f"{'='*70}")

    # Unified diff
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"BEFORE ({total} lines)",
        tofile=f"AFTER ({kept_count} lines)",
        lineterm="",
        n=1,  # 1 line of context
    )

    diff_lines = list(diff)
    if not diff_lines:
        print("  (no changes — kept everything)")
    else:
        # Show first 80 diff lines
        for line in diff_lines[:80]:
            print(line)
        if len(diff_lines) > 80:
            print(f"  ... ({len(diff_lines) - 80} more diff lines)")

    # Open in Cursor if requested
    if open_cursor and os.path.exists(CURSOR_CLI):
        with tempfile.NamedTemporaryFile(mode="w", suffix="_BEFORE.txt", delete=False, prefix=f"compact_{rec_id}_") as bf:
            bf.write(code)
            before_path = bf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix="_AFTER.txt", delete=False, prefix=f"compact_{rec_id}_") as af:
            af.write("\n".join(after_lines))
            after_path = af.name
        subprocess.Popen([CURSOR_CLI, "--diff", before_path, after_path])
        print(f"  Opened in Cursor: {before_path} vs {after_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify compaction quality via unified diff")
    parser.add_argument("--last", type=int, default=1, help="Show last N records")
    parser.add_argument("--id", type=str, default=None, help="Show specific record ID")
    parser.add_argument("--cursor", action="store_true", help="Open diff in Cursor")
    parser.add_argument("--output", type=str, default=OUTPUT, help="Path to output.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        print(f"No output file at {args.output}")
        sys.exit(1)

    records = load_records(args.output, last_n=args.last, record_id=args.id)
    if not records:
        print("No matching records found")
        sys.exit(1)

    for rec in records:
        show_diff(rec, open_cursor=args.cursor)

    print(f"\n--- {len(records)} record(s) shown ---")


if __name__ == "__main__":
    main()
