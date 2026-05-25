"""
ILM single-turn infilling — simplest version, trimmed output.


"""

import os
import re
import sys
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import GPT2LMHeadModel

ILM_SRC = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\ilm-master"
if ILM_SRC not in sys.path and os.path.exists(ILM_SRC):
    sys.path.insert(0, ILM_SRC)

import ilm.tokenize_util
from ilm.infer import infill_with_ilm


# =========================
# CONFIG
# =========================
IN_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\data_generation\event_keywords\generated_datasets"
OUT_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\ilm_infill_predictions"

MODEL_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\models\sto_ilm"
TOKENIZER = ilm.tokenize_util.Tokenizer.GPT2

GLOB_PATTERN = "*.json"
OUT_SUFFIX = "_ilm_infill.json"

NUM_INFILLS = 2

SKIP_IF_OUTPUT_EXISTS = False
STRIP_ASTERISKS = True

# Speaker prefixes used in the prompt — the trim looks for the LAST occurrence
# of any of these and keeps everything from there to the end.
SPEAKER_PREFIXES = ("User A:", "User B:", "User C:")


# =========================
# IO
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
    raise ValueError("Unsupported JSON shape.")


# =========================
# Field extraction
# =========================
def extract_summary(obj: Dict[str, Any]) -> str:
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


def format_mobile_context(obj: Dict[str, Any]) -> str:
    mcs = obj.get("mobile_context_snapshot")
    if not isinstance(mcs, dict):
        return ""
    lines = []
    for user_key in sorted(mcs.keys()):
        info = mcs.get(user_key)
        if not isinstance(info, dict):
            continue
        loc = info.get("location_semantic", "")
        gps = info.get("gps_coords", "")
        wifi = info.get("wifi_ssid", "")
        cal = info.get("calendar_next", "")
        parts = []
        if loc: parts.append(f"location={loc}")
        if gps: parts.append(f"gps={gps}")
        if wifi: parts.append(f"wifi={wifi}")
        if cal: parts.append(f"next_event={cal}")
        if parts:
            lines.append(f"{user_key}: " + "; ".join(parts))
    return "\n".join(lines)


def normalize_transcript(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = obj.get("conversation_transcript")
    if not isinstance(turns, list):
        return []
    out = []
    for i, t in enumerate(turns, start=1):
        if not isinstance(t, dict):
            continue
        tid = t.get("turn_id", i)
        try:
            tid = int(tid)
        except Exception:
            tid = i
        spk = str(t.get("speaker", "Unknown"))
        txt = str(t.get("text", ""))
        if STRIP_ASTERISKS:
            txt = txt.replace("*", "")
        out.append({"turn_id": tid, "speaker": spk, "text": txt})
    return out


def group_resolutions_by_turn(obj: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    resolutions = obj.get("ground_truth_resolutions") or []
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for r in resolutions:
        if not isinstance(r, dict):
            continue
        try:
            tid = int(r.get("trigger_turn_id"))
        except Exception:
            continue
        grouped.setdefault(tid, []).append(r)
    return grouped


# =========================
# Masking
# =========================
def mask_phrases(text: str, phrases: List[str]) -> Tuple[str, int]:
    masked = text
    installed = 0
    for phrase in phrases:
        if not isinstance(phrase, str) or not phrase:
            continue
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        new_masked, n = pattern.subn(" _", masked, count=1)
        if n > 0:
            masked = new_masked
            installed += 1
    masked = re.sub(r" {2,}", " ", masked).strip()
    return masked, installed


# =========================
# Prompt
# =========================
def build_prompt(summary: str, mobile_context: str, masked_turn: Dict[str, Any]) -> str:
    lines = []
    if summary:
        lines.append(summary)
    if mobile_context:
        lines.append(mobile_context)
    lines.append(f'{masked_turn["speaker"]}: "{masked_turn["text"]}"')
    return "\n".join(lines)


def install_infill_tokens(
    context_ids: List[int],
    blank_token_id: int,
    infill_word_id: int,
    expected_count: int,
) -> List[int]:
    out = list(context_ids)
    for _ in range(expected_count):
        try:
            idx = out.index(blank_token_id)
        except ValueError:
            break
        out[idx] = infill_word_id
    return out


# =========================
# Trim decoded text to just the speaker turn line
# =========================
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def trim_to_last_turn_line(decoded_text: str) -> str:
    """
    Find the LAST occurrence of any speaker prefix ('User A:', 'User B:', ...)
    in the decoded text and return from there to the end. Strip any ILM
    special tokens (<|...|>) from the result so only natural text remains.
    """
    # Find the last speaker prefix position
    last_idx = -1
    for prefix in SPEAKER_PREFIXES:
        idx = decoded_text.rfind(prefix)
        if idx > last_idx:
            last_idx = idx

    if last_idx == -1:
        # Couldn't find a speaker prefix; return the whole thing cleaned
        cleaned = SPECIAL_TOKEN_RE.sub("", decoded_text).strip()
        return cleaned

    tail = decoded_text[last_idx:]
    cleaned = SPECIAL_TOKEN_RE.sub("", tail).strip()
    # Collapse repeated whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# =========================
# Per-turn inference
# =========================
def run_infill_for_turn(
    model: GPT2LMHeadModel,
    additional_tokens_to_ids: Dict[str, int],
    tokenizer,
    summary: str,
    mobile_context: str,
    masked_turn: Dict[str, Any],
    num_infills: int,
) -> List[str]:
    prompt_text = build_prompt(summary, mobile_context, masked_turn)
    context_ids = ilm.tokenize_util.encode(prompt_text, tokenizer)

    blank_ids = ilm.tokenize_util.encode(' _', tokenizer)
    if not blank_ids:
        return []
    blank_token_id = blank_ids[0]
    infill_word_id = additional_tokens_to_ids['<|infill_word|>']

    n_blanks = context_ids.count(blank_token_id)
    if n_blanks == 0:
        return []

    new_ids = install_infill_tokens(context_ids, blank_token_id, infill_word_id, n_blanks)
    generated = infill_with_ilm(model, additional_tokens_to_ids, new_ids, num_infills=num_infills)

    out: List[str] = []
    for g in generated:
        raw = ilm.tokenize_util.decode(g, tokenizer)
        out.append(trim_to_last_turn_line(raw))
    return out


# =========================
# Per-file pipeline
# =========================
def process_file(
    model: GPT2LMHeadModel,
    additional_tokens_to_ids: Dict[str, int],
    tokenizer,
    fp: Path,
    out_path: Path,
) -> Dict[str, Any]:
    out_file = out_path / (fp.stem + OUT_SUFFIX)
    if SKIP_IF_OUTPUT_EXISTS and out_file.exists():
        return {"file": fp.name, "skipped": True}

    raw = load_json(fp)
    obj = unwrap_top_level(raw)

    summary = extract_summary(obj)
    mobile_context = format_mobile_context(obj)
    turns = normalize_transcript(obj)
    if not turns:
        raise ValueError("No conversation transcript found.")

    resolutions_by_turn = group_resolutions_by_turn(obj)
    if not resolutions_by_turn:
        save_json(out_file, {"file": fp.name, "results": []})
        return {"file": fp.name, "skipped": False, "num_targets": 0}

    idx_by_id = {t["turn_id"]: i for i, t in enumerate(turns)}
    results: List[Dict[str, Any]] = []

    for tid in sorted(resolutions_by_turn.keys()):
        if tid not in idx_by_id:
            continue
        target_turn = turns[idx_by_id[tid]]
        ground_truth_text = target_turn["text"]

        resolutions = resolutions_by_turn[tid]
        phrases = [r.get("ambiguous_phrase", "") for r in resolutions]

        masked_text, n_installed = mask_phrases(target_turn["text"], phrases)

        if n_installed == 0:
            results.append({
                "ground_truth_text": f'{target_turn["speaker"]}: "{ground_truth_text}"',
                "decoded_predictions": [],
            })
            continue

        masked_turn = {**target_turn, "text": masked_text}

        predictions = run_infill_for_turn(
            model=model,
            additional_tokens_to_ids=additional_tokens_to_ids,
            tokenizer=tokenizer,
            summary=summary,
            mobile_context=mobile_context,
            masked_turn=masked_turn,
            num_infills=NUM_INFILLS,
        )

        results.append({
            "ground_truth_text": f'{target_turn["speaker"]}: "{ground_truth_text}"',
            "decoded_predictions": predictions,
        })

    save_json(out_file, {"file": fp.name, "results": results})
    return {"file": fp.name, "skipped": False, "num_targets": len(results)}


# =========================
# Main batch
# =========================
def run_batch(
    in_dir: str = IN_DIR,
    out_dir: str = OUT_DIR,
    glob_pattern: str = GLOB_PATTERN,
) -> Dict[str, Any]:
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
        additional_ids_to_tokens = pickle.load(f)
    additional_tokens_to_ids = {v: k for k, v in additional_ids_to_tokens.items()}
    try:
        ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, TOKENIZER)
    except ValueError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {MODEL_DIR} (device={device})")
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    _ = model.to(device)

    files = sorted(in_path.glob(glob_pattern))
    if not files:
        raise RuntimeError(f"No files matched {glob_pattern} in {in_path}")

    summary = {"num_files": len(files), "processed": 0, "skipped": 0, "failed": 0, "failures": []}

    for fp in files:
        try:
            result = process_file(
                model=model,
                additional_tokens_to_ids=additional_tokens_to_ids,
                tokenizer=TOKENIZER,
                fp=fp,
                out_path=out_path,
            )
            if result.get("skipped"):
                summary["skipped"] += 1
                print(f"{fp.name} -> SKIPPED")
            else:
                summary["processed"] += 1
                print(f"{fp.name} -> {result.get('num_targets')} targets")
        except Exception as e:
            summary["failed"] += 1
            summary["failures"].append({"file": fp.name, "error": str(e)})
            print(f"{fp.name} -> FAILED: {e}")

    save_json(out_path / "batch_summary.json", summary)
    return summary


if __name__ == "__main__":
    s = run_batch()
    print(json.dumps(s, indent=2))