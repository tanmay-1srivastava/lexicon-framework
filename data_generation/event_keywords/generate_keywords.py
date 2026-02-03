#!/usr/bin/env python3
"""
Generate keywords from a CSV + prompt.txt using Azure OpenAI, and save output.

Usage:
  python3 generate_keywords.py \
    --csv /path/to/data.csv \
    --prompt /path/to/prompt.txt \
    --out /path/to/output.txt \
    --endpoint "https://initial-resources.openai.azure.com" \
    --deployment "gpt-4.1" \
    --api_version "2024-12-01-preview"

IMPORTANT:
- --endpoint must be the Azure OpenAI resource endpoint:
    https://<resource-name>.openai.azure.com
  NOT the cognitiveservices endpoint.

- --deployment must be the *deployment name* in Azure OpenAI Studio
  (often NOT the same as the model name).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

from openai import AzureOpenAI


def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="replace")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_messages(prompt_txt: str, csv_text: str) -> List[Dict[str, str]]:
    user_content = (
        "PROMPT (from prompt.txt):\n"
        f"{prompt_txt.strip()}\n\n"
        "CSV (from file):\n"
        "```csv\n"
        f"{csv_text.strip()}\n"
        "```\n"
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]


def load_security_key() -> str:
    """
    Loads security_key from:
      /Users/amartya/Documents/lexicon-framework/baselines/secret_keys.py
    """
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, type=str, help="Path to CSV file")
    parser.add_argument("--prompt", required=True, type=str, help="Path to prompt.txt")
    parser.add_argument("--out", required=True, type=str, help="Output file path to save model response")

    # Azure OpenAI config (make these REQUIRED; missing leads to confusing errors)
    parser.add_argument(
        "--endpoint",
        required=True,
        type=str,
        help='Azure OpenAI endpoint, e.g. "https://initial-resources.openai.azure.com"',
    )
    parser.add_argument(
        "--deployment",
        required=True,
        type=str,
        help="Azure OpenAI deployment name (exactly as in Azure OpenAI Studio)",
    )
    parser.add_argument("--api_version", default="2024-12-01-preview", type=str, help="Azure OpenAI API version")

    # Generation controls
    parser.add_argument("--max_completion_tokens", default=4096, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)

    # Token-safety: optionally truncate CSV text (characters) to avoid huge prompts
    parser.add_argument(
        "--max_csv_chars",
        default=300_000,
        type=int,
        help="Max CSV characters to include in the prompt (truncate if longer).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt.txt not found: {prompt_path}")

    endpoint = args.endpoint.strip().rstrip("/")
    if "openai.azure.com" not in endpoint:
        raise ValueError(
            f"--endpoint looks wrong: {endpoint}\n"
            "It must be like: https://<resource-name>.openai.azure.com\n"
            "NOT: https://<resource>.cognitiveservices.azure.com/"
        )

    # Load API key
    api_key = load_security_key()

    # Read inputs
    prompt_txt = read_text(prompt_path)
    csv_text = read_text(csv_path)

    if args.max_csv_chars is not None and len(csv_text) > args.max_csv_chars:
        csv_text = csv_text[: args.max_csv_chars] + "\n... (CSV truncated) ...\n"

    messages = build_messages(prompt_txt=prompt_txt, csv_text=csv_text)

    # Create client
    client = AzureOpenAI(
        api_key=api_key,
        api_version=args.api_version,
        azure_endpoint=endpoint,
    )

    # Call model (model=deployment name in Azure)
    resp = client.chat.completions.create(
        model=args.deployment,
        messages=messages,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    text_out = (resp.choices[0].message.content or "").strip()

    # Save output
    ensure_parent_dir(out_path)
    out_path.write_text(text_out + "\n", encoding="utf-8")
    print(f"[OK] Saved response to: {out_path}")


if __name__ == "__main__":
    main()
