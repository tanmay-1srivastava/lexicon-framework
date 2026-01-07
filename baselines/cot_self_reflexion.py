import os
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI
from secrets import azure_openai_key  # <-- your key string lives here


# =========================
# ✅ CONFIG (edit these)
# =========================
IN_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\data_generation\new_data\generated_datasets"
OUT_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\cot_self_reflexion"

AZURE_OPENAI_ENDPOINT = "https://initial-resources.cognitiveservices.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

GLOB_PATTERN = "*.json"
OUT_SUFFIX = "_cot_self_reflexionresult.json"

MAX_TOKENS = 4096
TEMPERATURE = 0.2
TOP_P = 1.0

# How many top-level keys to send to LLM (after unwrapping)
NUM_KEYS_TO_SEND = 2

# If we can't reliably take first N keys, fall back to this preference order
FALLBACK_KEY_ORDER = ["backstory", "mobile_context_snapshot", "conversation_transcript", "dataset_id"]


# =========================
# ✅ SYSTEM PROMPT (adds rationale + self_reflection, NOT chain-of-thought)
# =========================
SYSTEM_INSTRUCTIONS = """
You are an expert at resolving ambiguous references in dialogue using provided context.

CRITICAL:
- DO NOT reveal chain-of-thought or step-by-step reasoning.
- Output ONLY valid JSON (no markdown, no commentary).
- Your output MUST be a JSON object with a key named "ground_truth_resolutions".
- The value of "ground_truth_resolutions" MUST be a list of objects, each exactly of the form:

{
  "trigger_turn_id": int,
  "ambiguous_phrase": "",
  "resolved_entity": "",
  "resolution_source": "",
  "rationale": "",
  "self_reflection": {
    "confidence": "low|medium|high",
    "possible_failure_modes": ["", ""],
    "what_would_change_my_mind": ""
  }
}

Guidance for "rationale":
- Keep it short (1–2 sentences).
- Point to the most relevant evidence (e.g., a nearby turn, or a context field).
- DO NOT provide multi-step reasoning.

For "resolution_source", choose exactly one label from:
  "User A GPS + Wifi", "User B GPS + Wifi", "User A Calendar", "User B Calendar", "Conversation Context"
""".strip()


# =========================
# Strict JSON schema to force exact output shape
# =========================
OUTPUT_SCHEMA = {
    "name": "GroundTruthResolutionsWithRationaleOutput",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ground_truth_resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "trigger_turn_id": {"type": "integer"},
                        "ambiguous_phrase": {"type": "string"},
                        "resolved_entity": {"type": "string"},
                        "resolution_source": {"type": "string"},
                        "rationale": {"type": "string"},
                        "self_reflection": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "confidence": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                },
                                "possible_failure_modes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "what_would_change_my_mind": {"type": "string"},
                            },
                            "required": [
                                "confidence",
                                "possible_failure_modes",
                                "what_would_change_my_mind",
                            ],
                        },
                    },
                    "required": [
                        "trigger_turn_id",
                        "ambiguous_phrase",
                        "resolved_entity",
                        "resolution_source",
                        "rationale",
                        "self_reflection",
                    ],
                },
            }
        },
        "required": ["ground_truth_resolutions"],
    },
}


# =========================
# Helpers: IO
# =========================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# Helpers: unwrap top-level
# =========================
def unwrap_top_level(obj: Any) -> Dict[str, Any]:
    """
    Some datasets might be:
      - dict (normal)
      - list where the "real" object is the first element (no top-level key name)
    This returns the canonical dict object.
    """
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict):
            return first

    raise ValueError("Unsupported JSON shape: expected dict or list-of-dict with first element as dict.")


# =========================
# Helpers: transcript parsing
# =========================
def normalize_transcript(obj: Any) -> List[Dict[str, Any]]:
    """
    Output normalized shape:
      [{"turn_id": int, "speaker": str, "text": str}, ...]
    Accepts:
      - obj["conversation_transcript"]
      - obj["messages"] / obj["conversation"] / obj["turns"]
      - obj as list
    """
    if isinstance(obj, dict) and isinstance(obj.get("conversation_transcript"), list):
        return _coerce_turn_list(obj["conversation_transcript"])

    for key in ("messages", "conversation", "turns"):
        if isinstance(obj, dict) and isinstance(obj.get(key), list):
            return _coerce_turn_list(obj[key])

    if isinstance(obj, list):
        return _coerce_turn_list(obj)

    return []

def _coerce_turn_list(turns: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(turns, start=1):
        if not isinstance(t, dict):
            continue

        turn_id = t.get("turn_id") or t.get("id") or t.get("turn") or i
        speaker = t.get("speaker") or t.get("role") or t.get("from") or "Unknown"
        text = t.get("text") or t.get("content") or t.get("message") or ""

        try:
            turn_id = int(turn_id)
        except Exception:
            turn_id = i

        out.append({"turn_id": turn_id, "speaker": str(speaker), "text": str(text)})
    return out


# =========================
# Extract GT targets (optional)
# =========================
def extract_ground_truth_targets(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    If ground_truth_resolutions exists, return targets:
      [{"trigger_turn_id": int, "ambiguous_phrase": str}, ...]
    """
    gtr = obj.get("ground_truth_resolutions")
    if not isinstance(gtr, list) or not gtr:
        return None

    targets: List[Dict[str, Any]] = []
    for item in gtr:
        if not isinstance(item, dict):
            continue
        if "trigger_turn_id" in item and "ambiguous_phrase" in item:
            try:
                tid = int(item["trigger_turn_id"])
            except Exception:
                continue
            phrase = str(item["ambiguous_phrase"])
            targets.append({"trigger_turn_id": tid, "ambiguous_phrase": phrase})

    return targets if targets else None


# =========================
# ✅ Build "only first N keys" payload
# =========================
def pick_first_n_keys(d: Dict[str, Any], n: int) -> List[str]:
    keys = list(d.keys())
    if len(keys) >= n:
        return keys[:n]
    return keys

def build_llm_payload_only_first_keys(
    top_obj: Dict[str, Any],
    transcript_norm: List[Dict[str, Any]],
    n_keys: int = 2,
) -> Dict[str, Any]:
    """
    Build payload that contains ONLY the first n_keys top-level keys (in file order),
    BUT:
      - if 'conversation_transcript' is included, replace it with normalized transcript
      - never include ground_truth_resolutions (avoid leaking GT)
    If first n keys are unusable (e.g. missing transcript), fallback to preferred keys.
    """
    first_keys = [k for k in pick_first_n_keys(top_obj, n_keys) if k != "ground_truth_resolutions"]

    payload: Dict[str, Any] = {}
    for k in first_keys:
        if k == "conversation_transcript":
            payload[k] = transcript_norm
        else:
            payload[k] = top_obj.get(k)

    if "conversation_transcript" not in payload and transcript_norm:
        if len(payload) >= n_keys:
            last_key = list(payload.keys())[-1]
            payload.pop(last_key, None)
        payload["conversation_transcript"] = transcript_norm

    if not payload:
        fallback = []
        for k in FALLBACK_KEY_ORDER:
            if k in top_obj and k != "ground_truth_resolutions":
                fallback.append(k)
            if len(fallback) == n_keys:
                break

        payload = {}
        for k in fallback:
            if k == "conversation_transcript":
                payload[k] = transcript_norm
            else:
                payload[k] = top_obj.get(k)

    for k in list(payload.keys()):
        if payload[k] is None:
            payload[k] = ""

    return payload


# =========================
# Azure call (with retries)
# =========================
def call_azure_chat(
    client: AzureOpenAI,
    deployment: str,
    user_payload: Dict[str, Any],
    targets: Optional[List[Dict[str, Any]]],
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 1.0,
    retries: int = 5,
) -> Dict[str, Any]:
    """
    Returns parsed JSON object that MUST contain:
      {"ground_truth_resolutions": [...]}
    """

    if targets:
        user_task = {
            "task": "Resolve ONLY the given ambiguous phrases for the specified trigger_turn_id values.",
            "targets": targets,
            "instructions": [
                "For each target: find the turn text by trigger_turn_id and resolve the ambiguous_phrase.",
                "resolved_entity must be concrete (location/person/object/time).",
                "resolution_source must be one of: User A GPS + Wifi, User B GPS + Wifi, User A Calendar, User B Calendar, Conversation Context",
                "Add rationale (1–2 sentences) WITHOUT step-by-step reasoning.",
                "Add self_reflection with confidence + failure modes + what would change your mind.",
                "Return JSON with key ground_truth_resolutions (list of objects).",
            ],
            "input": user_payload,
        }
        mode = "guided_by_ground_truth"
    else:
        user_task = {
            "task": "Detect ambiguous phrases in the conversation (here/there/it/that/she/he/later/up/down/etc.) and resolve them.",
            "instructions": [
                "Return only genuinely ambiguous phrases that require context.",
                "Prefer high precision: do not add non-ambiguous phrases.",
                "resolution_source must be one of: User A GPS + Wifi, User B GPS + Wifi, User A Calendar, User B Calendar, Conversation Context",
                "Add rationale (1–2 sentences) WITHOUT step-by-step reasoning.",
                "Add self_reflection with confidence + failure modes + what would change your mind.",
                "Return JSON with key ground_truth_resolutions (list of objects).",
            ],
            "input": user_payload,
        }
        mode = "auto_detect"

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(user_task, ensure_ascii=False)},
    ]

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                response_format={"type": "json_schema", "json_schema": OUTPUT_SCHEMA},
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)

            if "ground_truth_resolutions" not in parsed or not isinstance(parsed["ground_truth_resolutions"], list):
                raise ValueError("Model output missing 'ground_truth_resolutions' list.")

            return {"mode": mode, "raw": content, "json": parsed}

        except Exception as e:
            last_err = e
            time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))

    raise RuntimeError(f"Azure call failed after {retries} retries. Last error: {last_err}")


# =========================
# Main batch function
# =========================
def run_batch(
    in_dir: str = IN_DIR,
    out_dir: str = OUT_DIR,
    glob_pattern: str = GLOB_PATTERN,
    out_suffix: str = OUT_SUFFIX,
) -> Dict[str, Any]:
    subscription_key = azure_openai_key
    if not subscription_key or not isinstance(subscription_key, str):
        raise RuntimeError("azure_openai_key is missing or not a string in secrets.py")

    client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=subscription_key,
    )

    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob(glob_pattern))
    if not files:
        raise RuntimeError(f"No files matched {glob_pattern} in {in_path}")

    summary = {
        "in_dir": str(in_path),
        "out_dir": str(out_path),
        "num_files": len(files),
        "processed": 0,
        "failed": 0,
        "failures": [],
    }

    for fp in files:
        try:
            raw = load_json(fp)
            obj = unwrap_top_level(raw)

            if not isinstance(obj, dict):
                raise ValueError("Top-level (or unwrapped) JSON is not an object/dict.")

            transcript_norm = normalize_transcript(obj)
            if not transcript_norm:
                raise ValueError("Could not find/normalize conversation transcript.")

            targets = extract_ground_truth_targets(obj)  # may be None

            llm_payload = build_llm_payload_only_first_keys(
                top_obj=obj,
                transcript_norm=transcript_norm,
                n_keys=NUM_KEYS_TO_SEND,
            )

            model_out = call_azure_chat(
                client=client,
                deployment=AZURE_OPENAI_DEPLOYMENT,
                user_payload=llm_payload,
                targets=targets,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            out_obj = {
                "dataset_id": obj.get("dataset_id", ""),
                "file": fp.name,
                "mode": model_out["mode"],
                "llm_input_keys_sent": list(llm_payload.keys()),
                "ground_truth_resolutions": model_out["json"]["ground_truth_resolutions"],
            }

            save_json(out_path / (fp.stem + out_suffix), out_obj)
            summary["processed"] += 1

        except Exception as e:
            summary["failed"] += 1
            summary["failures"].append({"file": fp.name, "error": str(e)})

    save_json(out_path / "batch_summary.json", summary)
    return summary


# =========================
# Entry
# =========================
if __name__ == "__main__":
    s = run_batch()
    print(json.dumps(s, indent=2))
