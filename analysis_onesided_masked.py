import json
import time
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI
from secrets import azure_openai_key  # <-- your key string lives here

from datetime import datetime, timezone


# =========================
# CONFIG (edit these)
# =========================
# Directory of input JSON files
IN_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\data_generation\event_keywords\generated_datasets"
OUT_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\single_turn_predictions"

AZURE_OPENAI_ENDPOINT = "https://initial-resources.cognitiveservices.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

GLOB_PATTERN = "*.json"

MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 1.0

# Mask both speakers, in this order: first User B, then User A.
# Each speaker produces its own output file (suffix differs per speaker).
MASK_SPEAKERS: List[str] = ["User B", "User A"]

# Per-speaker output suffix. Keys must match entries in MASK_SPEAKERS.
OUT_SUFFIX_BY_SPEAKER: Dict[str, str] = {
    "User B": "_single_turn_predictions_userB.json",
    "User A": "_single_turn_predictions_userA.json",
}

# Whether to include the masked-word-count hint ([MASKED - N words]) in the prompt
INCLUDE_TURN_LENGTH_HINT = True

# Stride between windows (1 = every consecutive window, 2 = every other, etc.)
WINDOW_STRIDE = 1

# Max number of windows to process per (file, speaker) pass (None = all)
MAX_WINDOWS: Optional[int] = None

# Skip a (file, speaker) pass if the output file already exists.
# Useful for resuming after a crash without redoing work.
SKIP_IF_OUTPUT_EXISTS = True


# =========================
# SYSTEM / TASK PROMPT
# =========================
# Note the explicit instruction (#9) to USE THE SUMMARY when making the prediction.
SYSTEM_PROMPT_TEMPLATE = """DIALOGUE COMPLETION TASK — {description}

CRITICAL INSTRUCTIONS FOR DIALOGUE COMPLETION:
1. PREDICT THE EXACT SYSTEM RESPONSE that would naturally follow in this conversation
2. PRESERVE ALL SPECIFIC DETAILS: times, dates, names, locations, numbers, reference codes, prices, phone numbers
3. ANTI-HALLUCINATION: Use 'XXXXXXX' for ALL specific information not available in the context that you need to provide (names, numbers, addresses, phone numbers, prices, times, etc.)
4. Maintain the same information density and factual accuracy as expected
5. Match the tone and style of the conversation
6. Include exact facts and specific information with XXXXXXX when relevant
7. Focus on providing the most relevant and complete information
8. You may use future turns (after the prediction turn) as background context to improve accuracy, but you must NOT explicitly include, mention, or preempt any new facts, topics, or requests that appear only in those future turns in your actual prediction.
9. USE THE CONVERSATION SUMMARY ABOVE as essential grounding for your prediction. The summary establishes the participants, their relationship, the setting, the agenda, and the goals of the conversation. Your predicted turn MUST be consistent with the summary — speaker roles, topics discussed, deadlines, named entities, and overall purpose should all align with what the summary describes. Do not produce a turn that contradicts the summary or drifts off-topic from it.

TASK: You are predicting what the speaker would say next in a natural conversation. Your response should be informative, specific, and helpful, AND it must be consistent with the conversation summary provided.
{turn_length_note}

EXAMPLE 1 - COMPLEX BOOKING:
User: I need accommodation
System: [MASKED - 7 words]
User: A hotel with parking
→ [PREDICTED] System: I found hotels with parking available. The XXXXXXX Hotel is available.
User: Book it for next weekend

EXAMPLE 2 - RESTAURANT RESERVATION:
User: I want to eat out tonight
System: [MASKED - 5 words]
User: Something expensive in the west
→ [PREDICTED] System: I found expensive restaurants in the west area available tonight.
User: Make a reservation for 8pm

EXAMPLE 3 - TRANSPORT BOOKING:
User: I need a train
System: [MASKED - 6 words]
User: To London on Friday
→ [PREDICTED] System: I have trains to London available on Friday. What time?
User: Book for 2 people

Output ONLY the predicted text for the [MASKED] turn — no labels, no quotes, no commentary."""

TURN_LENGTH_NOTE = "NOTE: [MASKED - n words] indicates the expected length of the response."


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


def unwrap_top_level(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    raise ValueError("Unsupported JSON shape: expected dict or list-of-dict.")


# =========================
# Helpers: transcript parsing
# =========================
def normalize_transcript(obj: Any) -> List[Dict[str, Any]]:
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
# Helpers: summary / description
# =========================
def extract_summary(obj: Dict[str, Any]) -> str:
    """
    Pull the best-available 'summary' / scenario description from the dataset.
    Priority: backstory.summary -> backstory (str) -> scenario_request -> ''
    """
    bs = obj.get("backstory")
    if isinstance(bs, dict):
        s = bs.get("summary")
        if isinstance(s, str) and s.strip():
            return s.strip()
    if isinstance(bs, str) and bs.strip():
        return bs.strip()

    gm = obj.get("generation_metadata")
    if isinstance(gm, dict):
        sr = gm.get("scenario_request")
        if isinstance(sr, str) and sr.strip():
            return sr.strip()

    sr_top = obj.get("scenario_request")
    if isinstance(sr_top, str) and sr_top.strip():
        return sr_top.strip()

    return ""


# =========================
# Helpers: windowing & prompt building
# =========================
def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def build_windows(
    turns: List[Dict[str, Any]],
    mask_speaker: str,
    stride: int = 1,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """
    Build 3-turn windows (prev, target, next) where target.speaker == mask_speaker.
    """
    windows = []
    i = 0
    while i <= len(turns) - 3:
        prev_t, tgt_t, next_t = turns[i], turns[i + 1], turns[i + 2]
        if tgt_t.get("speaker") == mask_speaker:
            windows.append((prev_t, tgt_t, next_t))
        i += stride
    return windows


def build_user_message(
    summary: str,
    prev_turn: Dict[str, Any],
    target_turn: Dict[str, Any],
    next_turn: Dict[str, Any],
    include_turn_length_hint: bool,
) -> str:
    """
    Build the user message containing summary + 3 turns with the middle one masked.
    """
    if include_turn_length_hint:
        masked_marker = f"[MASKED - {count_words(target_turn.get('text', ''))} words]"
    else:
        masked_marker = "[MASKED]"

    lines = []
    lines.append("CONVERSATION SUMMARY (use this to ground your prediction):")
    lines.append(summary if summary else "(no summary provided)")
    lines.append("")
    lines.append("CONVERSATION TURNS (predict the [MASKED] turn — it must be consistent with the summary above):")
    lines.append(f"{prev_turn.get('speaker', 'User A')}: {prev_turn.get('text', '')}")
    lines.append(f"{target_turn.get('speaker', 'User B')}: {masked_marker}")
    lines.append(f"{next_turn.get('speaker', 'User A')}: {next_turn.get('text', '')}")
    lines.append("")
    lines.append("Now output ONLY the predicted text for the [MASKED] turn.")
    return "\n".join(lines)


# =========================
# Azure call (with retries)
# =========================
def call_azure_chat(
    client: AzureOpenAI,
    deployment: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
    retries: int = 5,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
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
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            last_err = e
            time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))

    raise RuntimeError(f"Azure call failed after {retries} retries. Last error: {last_err}")


# =========================
# One pass: single file × single masked speaker
# =========================
def process_file_for_speaker(
    client: AzureOpenAI,
    fp: Path,
    out_path: Path,
    mask_speaker: str,
    out_suffix: str,
) -> Dict[str, Any]:
    out_file = out_path / (fp.stem + out_suffix)

    if SKIP_IF_OUTPUT_EXISTS and out_file.exists():
        return {
            "input_file": str(fp),
            "output_file": str(out_file),
            "mask_speaker": mask_speaker,
            "num_predictions": None,
            "skipped": True,
            "reason": "output_already_exists",
        }

    raw = load_json(fp)
    obj = unwrap_top_level(raw)
    if not isinstance(obj, dict):
        raise ValueError("Top-level (or unwrapped) JSON is not an object/dict.")

    turns = normalize_transcript(obj)
    if not turns:
        raise ValueError("Could not find/normalize conversation transcript.")

    description = extract_summary(obj)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        description=description if description else "(no description provided)",
        turn_length_note=(TURN_LENGTH_NOTE if INCLUDE_TURN_LENGTH_HINT else ""),
    )

    windows = build_windows(
        turns=turns,
        mask_speaker=mask_speaker,
        stride=WINDOW_STRIDE,
    )

    if MAX_WINDOWS is not None:
        windows = windows[:MAX_WINDOWS]

    predictions = []
    generated_at = datetime.now(timezone.utc).isoformat()

    for (prev_t, tgt_t, next_t) in windows:
        user_msg = build_user_message(
            summary=description,
            prev_turn=prev_t,
            target_turn=tgt_t,
            next_turn=next_t,
            include_turn_length_hint=INCLUDE_TURN_LENGTH_HINT,
        )

        pred_text = call_azure_chat(
            client=client,
            deployment=AZURE_OPENAI_DEPLOYMENT,
            system_prompt=system_prompt,
            user_message=user_msg,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        predictions.append({
            "masked_turn_id": tgt_t.get("turn_id"),
            "masked_speaker": tgt_t.get("speaker"),
            "prev_turn": {
                "turn_id": prev_t.get("turn_id"),
                "speaker": prev_t.get("speaker"),
                "text": prev_t.get("text"),
            },
            "next_turn": {
                "turn_id": next_t.get("turn_id"),
                "speaker": next_t.get("speaker"),
                "text": next_t.get("text"),
            },
            "ground_truth_text": tgt_t.get("text"),
            "ground_truth_word_count": count_words(tgt_t.get("text", "")),
            "predicted_text": pred_text,
            "predicted_word_count": count_words(pred_text),
        })

    out_obj = {
        "dataset_id": obj.get("dataset_id", ""),
        "file": fp.name,
        "summary_used": description,
        "mask_speaker": mask_speaker,
        "include_turn_length_hint": INCLUDE_TURN_LENGTH_HINT,
        "num_predictions": len(predictions),
        "predictions": predictions,
        "generation_metadata": {
            "model": AZURE_OPENAI_DEPLOYMENT,
            "temperature": float(TEMPERATURE),
            "top_p": float(TOP_P),
            "max_tokens": MAX_TOKENS,
            "generated_at": generated_at,
            "provider": "Azure OpenAI",
            "endpoint": AZURE_OPENAI_ENDPOINT,
        },
    }

    save_json(out_file, out_obj)

    return {
        "input_file": str(fp),
        "output_file": str(out_file),
        "mask_speaker": mask_speaker,
        "num_predictions": len(predictions),
        "skipped": False,
    }


# =========================
# Main batch: ALL files × BOTH speakers (User B first, then User A)
# =========================
def run_batch(
    in_dir: str = IN_DIR,
    out_dir: str = OUT_DIR,
    glob_pattern: str = GLOB_PATTERN,
) -> Dict[str, Any]:
    if not azure_openai_key or not isinstance(azure_openai_key, str):
        raise RuntimeError("azure_openai_key is missing or not a string in secrets.py")

    client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=azure_openai_key,
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
        "mask_speakers_order": MASK_SPEAKERS,
        "passes": [],
        "totals": {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_predictions": 0,
        },
        "failures": [],
    }

    # ---- Outer loop: speakers (User B first, then User A) ----
    for mask_speaker in MASK_SPEAKERS:
        out_suffix = OUT_SUFFIX_BY_SPEAKER.get(mask_speaker)
        if not out_suffix:
            # Fallback if user adds a new speaker to MASK_SPEAKERS without updating the map
            safe_name = re.sub(r"\W+", "", mask_speaker).lower() or "speaker"
            out_suffix = f"_single_turn_predictions_{safe_name}.json"

        pass_info = {
            "mask_speaker": mask_speaker,
            "out_suffix": out_suffix,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_predictions": 0,
        }

        # ---- Inner loop: every file in the directory ----
        for fp in files:
            try:
                result = process_file_for_speaker(
                    client=client,
                    fp=fp,
                    out_path=out_path,
                    mask_speaker=mask_speaker,
                    out_suffix=out_suffix,
                )
                if result.get("skipped"):
                    pass_info["skipped"] += 1
                    summary["totals"]["skipped"] += 1
                else:
                    pass_info["processed"] += 1
                    pass_info["total_predictions"] += int(result.get("num_predictions") or 0)
                    summary["totals"]["processed"] += 1
                    summary["totals"]["total_predictions"] += int(result.get("num_predictions") or 0)

                print(f"[{mask_speaker}] {fp.name} -> "
                      f"{'SKIPPED' if result.get('skipped') else result.get('num_predictions')}"
                      f" preds")

            except Exception as e:
                pass_info["failed"] += 1
                summary["totals"]["failed"] += 1
                summary["failures"].append({
                    "mask_speaker": mask_speaker,
                    "file": fp.name,
                    "error": str(e),
                })
                print(f"[{mask_speaker}] {fp.name} -> FAILED: {e}")

        summary["passes"].append(pass_info)

    save_json(out_path / "batch_summary.json", summary)
    return summary


# =========================
# Entry
# =========================
if __name__ == "__main__":
    s = run_batch()
    print(json.dumps(s, indent=2))