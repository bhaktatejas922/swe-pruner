#!/usr/bin/env python3
"""Download and prepare diverse HF agent trace datasets for labeling.

Streams from HuggingFace, serializes to flat text with role markers,
windows into ~8K token chunks, writes input JSONL for the labeling pipeline.

Usage:
    python3 train/labeling/prepare_hf_agent_traces.py \
        --output train/data/labeling_tasks/input_hf_agents.jsonl \
        --max-per-source 5000 \
        --start-id 30000
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Reuse existing helpers
sys.path.insert(0, os.path.dirname(__file__))
from prepare_agent_traces import (
    _normalize_openai_format,
    _normalize_swe_agent_format,
    _trunc,
    _role_to_marker,
    serialize_conversation,
    extract_query,
    window_text,
    _is_mostly_tool_output,
)


# ─── Dataset-specific normalizers ────────────────────────────────────────────


def _normalize_toucan(sample: dict) -> Optional[List[Dict]]:
    """Normalize Toucan-1.5M: messages is a JSON string of OpenAI-format messages."""
    raw = sample.get("messages", "")
    if isinstance(raw, str):
        try:
            messages = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(raw, list):
        messages = raw
    else:
        return None
    if not messages or len(messages) < 2:
        return None
    return _normalize_openai_format(messages)


def _normalize_nebius_swe(sample: dict) -> Optional[List[Dict]]:
    """Normalize nebius/SWE-agent-trajectories: trajectory is list of {role, text, mask}."""
    trajectory = sample.get("trajectory", [])
    if not trajectory or not isinstance(trajectory, list) or len(trajectory) < 3:
        return None
    normalized = []
    for entry in trajectory:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "unknown")
        text = entry.get("text", entry.get("content", ""))
        normalized.append({"role": role, "content": _trunc(text)})
    return normalized if len(normalized) >= 3 else None


def _normalize_glaive(sample: dict) -> Optional[List[Dict]]:
    """Normalize glaiveai/glaive-function-calling-v2: system + chat fields."""
    system = sample.get("system", "")
    chat = sample.get("chat", "")
    if not chat:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": _trunc(system)})

    # Parse chat text: USER: ... ASSISTANT: ... FUNCTION RESPONSE: ...
    current_role = None
    current_content = []

    for line in chat.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("USER:"):
            if current_role:
                messages.append({"role": current_role, "content": _trunc("\n".join(current_content))})
            current_role = "user"
            current_content = [line_stripped[5:].strip()]
        elif line_stripped.startswith("ASSISTANT:"):
            if current_role:
                messages.append({"role": current_role, "content": _trunc("\n".join(current_content))})
            current_role = "assistant"
            current_content = [line_stripped[10:].strip()]
        elif line_stripped.startswith("FUNCTION RESPONSE:"):
            if current_role:
                messages.append({"role": current_role, "content": _trunc("\n".join(current_content))})
            current_role = "tool"
            current_content = [line_stripped[18:].strip()]
        elif line_stripped.startswith("<functioncall>"):
            # Part of assistant turn
            current_content.append(line_stripped)
        else:
            current_content.append(line)

    if current_role and current_content:
        messages.append({"role": current_role, "content": _trunc("\n".join(current_content))})

    return messages if len(messages) >= 2 else None


def _normalize_claude_traces(sample: dict) -> Optional[List[Dict]]:
    """Normalize nlile/misc-merged-claude-code-traces-v1: messages_json field."""
    raw = sample.get("messages_json", "")
    if not raw or raw == "null":
        return None
    try:
        messages = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not messages or not isinstance(messages, list) or len(messages) < 2:
        return None

    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle list content (Claude format with content blocks)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        parts.append(f"[calling {block.get('name', 'tool')}]: {json.dumps(block.get('input', {}))[:2000]}")
                    elif block.get("type") == "tool_result":
                        parts.append(str(block.get("content", ""))[:3000])
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)

        normalized.append({"role": role, "content": _trunc(content)})

    return normalized if len(normalized) >= 2 else None


# ─── Streaming + Processing ──────────────────────────────────────────────────


def process_dataset(
    dataset_name: str,
    config: Optional[str],
    normalizer,
    query_extractor,
    source_label: str,
    output_file,
    max_samples: int,
    start_id: int,
    window_lines: int = 800,
) -> int:
    """Stream a HF dataset, normalize, window, write to output."""
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}" + (f" [{config}]" if config else ""))

    kwargs = {"streaming": True, "split": "train"}
    if config:
        kwargs["name"] = config

    try:
        ds = load_dataset(dataset_name, **kwargs)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return 0

    count = 0
    skipped = 0
    current_id = start_id

    for sample in ds:
        if count >= max_samples:
            break

        # Normalize messages
        messages = normalizer(sample)
        if not messages or len(messages) < 2:
            skipped += 1
            continue

        # Extract query
        query = query_extractor(sample, messages)

        # Serialize
        serialized = serialize_conversation(messages, max_content_per_msg=3000)
        total_lines = len(serialized.split("\n"))
        if total_lines < 20:
            skipped += 1
            continue

        # Window
        windows = window_text(serialized, window_lines=window_lines)

        for window in windows:
            if count >= max_samples:
                break
            if _is_mostly_tool_output(window):
                skipped += 1
                continue

            record = {
                "id": f"{current_id:05d}",
                "query": query or f"Agent trace from {source_label}",
                "code": window,
                "source": source_label,
                "data_type": "agent_trace",
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            current_id += 1

        if count % 1000 == 0 and count > 0:
            print(f"  {count} windows written...")

    print(f"  Done: {count} windows, {skipped} skipped")
    return count


def _extract_query_toucan(sample, messages):
    """Toucan has a 'question' field."""
    q = sample.get("question", "")
    if q and len(q) > 10:
        return q[:500]
    return extract_query({}, messages)


def _extract_query_nebius(sample, messages):
    """Nebius has instance_id but query is in first user message."""
    return extract_query(sample, messages)


def _extract_query_glaive(sample, messages):
    """Glaive query is the first USER message."""
    return extract_query({}, messages)


def _extract_query_claude(sample, messages):
    """Claude traces: use user_prompt or first user message."""
    up = sample.get("user_prompt", "")
    if up and len(up.strip()) > 10:
        return up.strip()[:500]
    return extract_query({}, messages)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Download and prepare HF agent traces")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max-per-source", type=int, default=5000, help="Max samples per dataset")
    parser.add_argument("--start-id", type=int, default=30000, help="Starting ID number")
    parser.add_argument("--window-lines", type=int, default=800, help="Lines per window")
    parser.add_argument(
        "--datasets", type=str, default="toucan,nebius,glaive,claude",
        help="Comma-separated dataset names to process (toucan,nebius,glaive,claude)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    datasets_to_run = set(args.datasets.split(","))

    DATASETS = []
    if "toucan" in datasets_to_run:
        DATASETS.append(("Agent-Ark/Toucan-1.5M", "OSS", _normalize_toucan, _extract_query_toucan, "toucan_oss"))
    if "nebius" in datasets_to_run:
        DATASETS.append(("nebius/SWE-agent-trajectories", None, _normalize_nebius_swe, _extract_query_nebius, "nebius_swe_agent"))
    if "glaive" in datasets_to_run:
        DATASETS.append(("glaiveai/glaive-function-calling-v2", None, _normalize_glaive, _extract_query_glaive, "glaive_func_call"))
    if "claude" in datasets_to_run:
        DATASETS.append(("nlile/misc-merged-claude-code-traces-v1", None, _normalize_claude_traces, _extract_query_claude, "claude_code_traces"))

    total = 0
    current_id = args.start_id

    with open(args.output, "w") as out_f:
        for ds_name, config, normalizer, query_fn, source_label in DATASETS:
            n = process_dataset(
                dataset_name=ds_name,
                config=config,
                normalizer=normalizer,
                query_extractor=query_fn,
                source_label=source_label,
                output_file=out_f,
                max_samples=args.max_per_source,
                start_id=current_id,
                window_lines=args.window_lines,
            )
            total += n
            current_id += n

    print(f"\n{'='*60}")
    print(f"TOTAL: {total} windows -> {args.output}")
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f"Output size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
