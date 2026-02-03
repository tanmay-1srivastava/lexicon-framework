#!/usr/bin/env python3
"""
Generic Event Generator (Prompt + Keywords -> Events) using Azure OpenAI.

What it does
------------
- Reads:
    1) prompt.txt  (instructions / template)
    2) keywords.txt (one keyword/phrase per line OR comma-separated)
- Calls Azure OpenAI (chat completions)
- Saves:
    - a JSONL file (one event per line) OR a JSON file
    - optionally also saves raw model output

Expected output format (recommended)
-----------------------------------
This script asks the model to output STRICT JSONL lines like:
{"event":"...","keywords":["...","..."],"category":"...","severity":1,"rationale":"..."}  # rationale optional

Usage
-----
python3 generate_events.py \
  --prompt /path/to/prompt.txt \
  --keywords /path/to/keywords.txt \
  --out /path/to/events.jsonl \
  --endpoint "https://<resource>.openai.azure.com" \
  --deployment "<deployment-name>" \
  --api_version "2024-12-01-preview" \
  --n_events 50 \
  --seed 123 \
  --temperature 0.8

Notes
-----
- --deployment is the Azure OpenAI *deployment name*
- The API key must be in:
    /Users/amartya/Documents/lexicon-framework/baselines/secret_keys.py
  as: security_key = "..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI


# -------------------------
# IO helpers
# -------------------------
def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="replace")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_security_key() -> str:
    secret_dir = Path("/Users/amartya/Documents/lexicon-framework/baselines").resolve()
    if not secret_dir.exists():
        raise FileNotFoundError(f"Secret dir not found: {secret_dir}")
    if str(secret_dir) not in sys.path:
        sys.path.insert(0, str(secret_dir))

    try:
        from secret_keys import security_key  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import `security_key` from secret_keys.py. "
            "Ensure it contains: security_key = '...'"
        ) from e

    if not isinstance(security_key, str) or not security_key.strip():
        raise RuntimeError("security_key is missing/empty or not a string in secret_keys.py")

    return security_key.strip()


def parse_keywords(text: str) -> List[str]:
    """
    Accepts keywords in either:
      - one per line
      - comma-separated
      - bullet list lines like "- fever"
    Returns a cleaned, de-duplicated list preserving order.
    """
    raw = text.strip()

    # If commas are heavily used, split on commas; otherwise split by lines.
    if raw.count(",") >= 3 and raw.count("\n") <= 2:
        candidates = [x.strip() for x in raw.split(",")]
    else:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # strip bullets
        candidates = [re.sub(r"^\s*[-*•]\s*", "", ln).strip() for ln in lines]

    # clean + dedupe
    out: List[str] = []
    seen = set()
    for k in candidates:
        k2 = re.sub(r"\s+", " ", k).strip()
        if not k2:
            continue
        if k2.lower() in seen:
            continue
        seen.add(k2.lower())
        out.append(k2)
    return out


# -------------------------
# Prompting
# -------------------------
def build_messages(
    base_prompt: str,
    keywords: List[str],
    n_events: int,
    output_schema_hint: str,
) -> List[Dict[str, str]]:
    kw_block = "\n".join(f"- {k}" for k in keywords)

    user = f"""
You will generate synthetic "events" using the provided keywords.

BASE PROMPT / INSTRUCTIONS:
{base_prompt.strip()}

KEYWORDS (use these as anchors; you may combine multiple keywords in one event):
{kw_block}

TASK:
- Generate exactly {n_events} distinct events.
- Each event should be realistic, specific, and not repetitive.
- Prefer using 2–5 keywords per event when reasonable.
- Do NOT invent extra metadata fields beyond the schema.

OUTPUT FORMAT (STRICT):
{output_schema_hint}

IMPORTANT:
- Output ONLY in the required format. No extra commentary.
""".strip()

    return [
        {"role": "system", "content": "You are a careful data generator that follows formatting rules exactly."},
        {"role": "user", "content": user},
    ]


# -------------------------
# Output parsing
# -------------------------
def try_parse_jsonl(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Attempts to parse JSONL (one JSON object per line).
    Returns (parsed_objects, bad_lines)
    """
    objs: List[Dict[str, Any]] = []
    bad: List[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # Sometimes model wraps in ```jsonl ... ```
        if ln.startswith("```") or ln.endswith("```"):
            continue
        try:
            val = json.loads(ln)
            if isinstance(val, dict):
                objs.append(val)
            else:
                bad.append(ln)
        except Exception:
            bad.append(ln)
    return objs, bad


def normalize_events(
    objs: List[Dict[str, Any]],
    required_fields: List[str],
) -> List[Dict[str, Any]]:
    """
    Ensures each dict has required fields (if missing, set None).
    """
    out = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        for f in required_fields:
            o.setdefault(f, None)
        out.append(o)
    return out


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", required=True, help="Path to prompt.txt")
    parser.add_argument("--keywords", required=True, help="Path to keywords.txt")
    parser.add_argument("--out", required=True, help="Output path (.jsonl recommended)")
    parser.add_argument("--raw_out", default=None, help="Optional path to save raw model output")

    # Azure config
    parser.add_argument("--endpoint", required=True, help="https://<resource>.openai.azure.com")
    parser.add_argument("--deployment", required=True, help="Azure OpenAI deployment name")
    parser.add_argument("--api_version", default="2024-12-01-preview", help="Azure OpenAI API version")

    # generation
    parser.add_argument("--n_events", type=int, default=50, help="How many events to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_completion_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed (best-effort)")

    args = parser.parse_args()

    prompt_path = Path(args.prompt).expanduser().resolve()
    keywords_path = Path(args.keywords).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    raw_out_path = Path(args.raw_out).expanduser().resolve() if args.raw_out else None

    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt not found: {prompt_path}")
    if not keywords_path.exists():
        raise FileNotFoundError(f"keywords not found: {keywords_path}")

    endpoint = args.endpoint.strip().rstrip("/")
    if "openai.azure.com" not in endpoint:
        raise ValueError(
            f"--endpoint looks wrong: {endpoint}\n"
            "It must be like: https://<resource-name>.openai.azure.com"
        )

    base_prompt = read_text(prompt_path)
    kw_text = read_text(keywords_path)
    keywords = parse_keywords(kw_text)
    if not keywords:
        raise ValueError("No keywords parsed from keywords file.")

    # You can customize schema based on your project needs:
    # Keep it minimal and stable for downstream processing.
    output_schema_hint = """Return EXACTLY one JSON object per line (JSONL), with keys:
{"event": <string>, "keywords": <list of strings>, "category": <string>}
No surrounding list, no markdown fences, no extra text."""

    messages = build_messages(
        base_prompt=base_prompt,
        keywords=keywords,
        n_events=args.n_events,
        output_schema_hint=output_schema_hint,
    )

    api_key = load_security_key()
    client = AzureOpenAI(
        api_key=api_key,
        api_version=args.api_version,
        azure_endpoint=endpoint,
    )

    # Build request
    create_kwargs: Dict[str, Any] = dict(
        model=args.deployment,
        messages=messages,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Seed is supported in some OpenAI APIs; Azure support varies by version/model.
    # If unsupported, Azure will ignore or raise; we handle gracefully.
    if args.seed is not None:
        create_kwargs["seed"] = args.seed

    resp = client.chat.completions.create(**create_kwargs)
    text = (resp.choices[0].message.content or "").strip()

    if raw_out_path:
        ensure_parent_dir(raw_out_path)
        raw_out_path.write_text(text + "\n", encoding="utf-8")

    # Parse JSONL
    objs, bad_lines = try_parse_jsonl(text)

    # If parsing failed badly, save raw and raise a helpful error.
    if len(objs) == 0:
        # Save raw next to output to debug
        fallback_raw = out_path.with_suffix(out_path.suffix + ".raw.txt")
        ensure_parent_dir(fallback_raw)
        fallback_raw.write_text(text + "\n", encoding="utf-8")
        raise RuntimeError(
            "Model output could not be parsed as JSONL.\n"
            f"Saved raw output to: {fallback_raw}\n"
            "Tip: Lower temperature or strengthen formatting rules in prompt."
        )

    required_fields = ["event", "keywords", "category"]
    events = normalize_events(objs, required_fields=required_fields)

    # Ensure exactly n_events if possible (truncate extras)
    if len(events) > args.n_events:
        events = events[: args.n_events]

    # If fewer than requested, still save what we got, but warn.
    if len(events) < args.n_events:
        print(f"[WARN] Requested {args.n_events} events but parsed {len(events)} JSON objects.")
        if bad_lines:
            print(f"[WARN] Also had {len(bad_lines)} unparsable lines (saved in .bad_lines.txt).")

    ensure_parent_dir(out_path)

    # Write JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Save bad lines (if any) for debugging
    if bad_lines:
        bad_path = out_path.with_suffix(out_path.suffix + ".bad_lines.txt")
        bad_path.write_text("\n".join(bad_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote events JSONL: {out_path}")
    if raw_out_path:
        print(f"[OK] Wrote raw output: {raw_out_path}")


if __name__ == "__main__":
    main()
