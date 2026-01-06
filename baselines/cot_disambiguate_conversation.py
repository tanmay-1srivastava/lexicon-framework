"""
Disambiguate ambiguous phrases in a conversation JSON using Azure OpenAI.

✅ Corrected so you DON'T need to pass endpoint/deployment/api-key via CLI.
It will use values defined in THIS FILE (constants below) OR env vars if you prefer.

How secrets work (recommended):
- Put your key in environment variable AZURE_OPENAI_API_KEY
OR
- Put it in secret_keys.py as: security_key = "..."

Run:
  python3 baselines/cot_disambiguate_conversation.py \
    --input path/to/conversation.json \
    --output path/to/clarified.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openai import AzureOpenAI

# ----------------------------
# ✅ CONFIG (set once here)
# ----------------------------
AZURE_OPENAI_ENDPOINT = "https://initial-resources.cognitiveservices.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

# ✅ API key: prefer env var; fallback to secret_keys.py if present
try:
    from secret_keys import security_key  # local-only, do not commit
except Exception:
    security_key = ""

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip() or security_key


@dataclass
class AzureConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _speaker_to_role(speaker: str) -> str:
    s = (speaker or "").strip().lower()
    if s in {"assistant", "system", "user"}:
        return s
    if s in {"bot", "agent", "model"}:
        return "assistant"
    return "user"


def _normalize_messages(raw: Any) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Return (messages, envelope_metadata).
    messages is always a list of {role, content}.
    envelope_metadata carries any original wrapper fields for round-tripping.
    """
    envelope: Dict[str, Any] = {}

    if isinstance(raw, dict):
        envelope = {k: v for k, v in raw.items() if k not in {"messages", "conversation", "turns", "dialogue", "conversation_transcript"}}
        if isinstance(raw.get("messages"), list):
            raw_messages = raw["messages"]
        elif isinstance(raw.get("conversation"), list):
            raw_messages = raw["conversation"]
        elif isinstance(raw.get("turns"), list):
            raw_messages = raw["turns"]
        elif isinstance(raw.get("dialogue"), list):
            raw_messages = raw["dialogue"]
        elif isinstance(raw.get("conversation_transcript"), list):
            raw_messages = raw["conversation_transcript"]
        else:
            raise ValueError(
                "Unsupported JSON object shape. Expected one of keys: messages|conversation|turns|dialogue|conversation_transcript."
            )
    elif isinstance(raw, list):
        raw_messages = raw
    else:
        raise ValueError("Unsupported JSON root type. Expected object or list.")

    messages: List[Dict[str, str]] = []
    for i, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise ValueError(f"Turn {i} is not an object: {type(item)}")

        # Preferred: role/content
        if "role" in item and "content" in item:
            role = str(item.get("role", "user"))
            content = item.get("content", "")
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(part))
                content_str = "\n".join([p for p in parts if p])
            else:
                content_str = str(content)
            messages.append({"role": role, "content": content_str})
            continue

        # Alternate: speaker/text
        if "speaker" in item and ("text" in item or "content" in item):
            role = _speaker_to_role(str(item.get("speaker", "user")))
            content_str = str(item.get("text", item.get("content", "")))
            messages.append({"role": role, "content": content_str})
            continue

        # Alternate: from/message
        if "from" in item and "message" in item:
            role = _speaker_to_role(str(item.get("from", "user")))
            messages.append({"role": role, "content": str(item.get("message", ""))})
            continue

        raise ValueError(
            f"Unsupported message format at index {i}. Expected role/content or speaker/text. "
            f"Got keys: {sorted(item.keys())}"
        )

    return messages, envelope


def _extract_json_object(text: str) -> Any:
    """Try to parse a JSON object/array from model output."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        raise ValueError("Model output did not contain parseable JSON.")

    candidate = match.group(1)
    
    # Clean up common truncation artifacts
    candidate = re.sub(r'[%\x00]+$', '', candidate)  # Remove trailing junk
    candidate = re.sub(r'\}\}+(\]?)$', r'}\1', candidate)  # Fix double closing braces
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Try harder: find the last complete JSON object/array and truncate there
        depth_obj = 0
        depth_arr = 0
        last_valid_pos = 0
        
        for i, char in enumerate(candidate):
            if char == '{':
                depth_obj += 1
            elif char == '}':
                depth_obj -= 1
            elif char == '[':
                depth_arr += 1
            elif char == ']':
                depth_arr -= 1
            
            # Mark position where all depths are balanced and we're not in a string
            if depth_obj == 0 and depth_arr == 0 and char in '}]':
                last_valid_pos = i + 1
        
        if last_valid_pos > 0 and last_valid_pos < len(candidate):
            candidate = candidate[:last_valid_pos]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        
        # Debug: save the malformed JSON to a file
        debug_file = "debug_model_output.txt"
        with open(debug_file, "w") as f:
            f.write(candidate)
        print(f"DEBUG: Saved malformed model output to {debug_file}")
        print(f"DEBUG: Error: {e}")
        raise ValueError(f"Failed to parse extracted JSON from model output: {e}\nSee {debug_file} for details.")


def build_prompt(messages: List[Dict[str, str]]) -> str:
    instructions = (
        "You resolve ambiguous phrases in a conversation (pronouns, deixis like 'here/there', "
        "vague time like 'then/later/next week', underspecified references like 'that one').\n\n"
        "Important rules:\n"
        "- Use ONLY information present in the conversation. Do NOT invent new facts.\n"
        "- If a phrase cannot be confidently resolved, mark it needs_clarification=true and propose a clarifying_question.\n"
        "- For resolved_entity, provide DETAILED context including relevant facts from the conversation.\n"
        "- Think step-by-step internally, but DO NOT output your reasoning. Output ONLY valid JSON.\n\n"
        "Output JSON schema (must be valid JSON, no comments):\n"
        "{\n"
        "  \"clarified_messages\": [ {\"role\": \"user|assistant|system\", \"content\": \"...\"}, ... ],\n"
        "  \"resolutions\": [\n"
        "    {\n"
        "      \"trigger_turn_id\": 1,\n"
        "      \"ambiguous_phrase\": \"...\",\n"
        "      \"resolved_entity\": \"[detailed entity with context from conversation]\",\n"
        "      \"resolution_type\": \"person|object|spatial|temporal|other\",\n"
        "      \"confidence\": 0.0,\n"
        "      \"needs_clarification\": false,\n"
        "      \"clarifying_question\": null\n"
        "    }\n"
        "  ],\n"
        "  \"clarifying_questions\": [\"...\"]\n"
        "}\n\n"
        "Clarification format:\n"
        "- In clarified_messages, replace the ambiguous phrase with a clarified version that preserves meaning.\n"
        "  Example: 'Meet me there at 5' -> 'Meet me at the office at 5'.\n"
        "- For resolved_entity: provide specific details. Example: 'Dr. Patel (cardiologist at Evergreen Medical Center)' not just 'doctor'.\n"
    )

    convo_lines: List[str] = []
    for i, m in enumerate(messages):
        convo_lines.append(f"[{i}] {m.get('role','user')}: {m.get('content','')}")
    return instructions + "\nConversation:\n" + "\n".join(convo_lines)


def disambiguate(
    client: AzureOpenAI,
    deployment: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    prompt = build_prompt(messages)

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a careful disambiguation assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_completion_tokens=max_completion_tokens,
    )

    content = response.choices[0].message.content or ""
    parsed = _extract_json_object(content)

    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object.")

    clarified_messages = parsed.get("clarified_messages")
    if not isinstance(clarified_messages, list):
        raise ValueError("Model output missing clarified_messages (list).")

    resolutions = parsed.get("resolutions") or []
    clarifying_questions = parsed.get("clarifying_questions") or []

    return {
        "clarified_messages": clarified_messages,
        "resolutions": resolutions,
        "clarifying_questions": clarifying_questions,
        "model": deployment,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve ambiguous phrases in a conversation JSON via Azure OpenAI")
    p.add_argument("--input", required=True, help="Path to input conversation JSON")
    p.add_argument("--output", default=None, help="Path to write clarified JSON (optional)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--test", action="store_true", help="Run a simple Azure OpenAI test prompt and exit")
    return p.parse_args()


def _make_client(cfg: AzureConfig) -> AzureOpenAI:
    return AzureOpenAI(
        api_version=cfg.api_version,
        azure_endpoint=cfg.endpoint,
        api_key=cfg.api_key,
    )


def run_test(cfg: AzureConfig) -> None:
    client = _make_client(cfg)
    resp = client.chat.completions.create(
        model=cfg.deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        max_completion_tokens=300,
        temperature=0.7,
        top_p=1.0,
    )
    print(resp.choices[0].message.content or "")


def main() -> None:
    # ✅ Use config from THIS FILE
    cfg = AzureConfig(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment=AZURE_OPENAI_DEPLOYMENT,
    )

    if not cfg.endpoint:
        raise SystemExit("Missing endpoint in code: AZURE_OPENAI_ENDPOINT")
    if not cfg.deployment:
        raise SystemExit("Missing deployment in code: AZURE_OPENAI_DEPLOYMENT")
    if not cfg.api_key:
        raise SystemExit(
            "Missing API key. Set env var AZURE_OPENAI_API_KEY or create secret_keys.py with security_key."
        )

    args = parse_args()

    if args.test:
        run_test(cfg)
        return

    raw = _read_json(args.input)
    messages, envelope = _normalize_messages(raw)

    client = _make_client(cfg)

    result = disambiguate(
        client,
        cfg.deployment,
        messages,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    # Auto-generate output path if not provided
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_filename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(script_dir, f"{input_filename}_cot_result.json")

    output_obj: Dict[str, Any] = result["resolutions"]

    _write_json(args.output, output_obj)
    print(f"Wrote clarified conversation to: {args.output}")


if __name__ == "__main__":
    main()
