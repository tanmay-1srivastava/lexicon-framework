import json
import os
import random
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import requests


# ============================================================
# CONFIG
# ============================================================

# --- Reasoning mode ---
# Set it here (True = thinking on, False = off) ...
THINK = False
# ... or override it from SLURM:  sbatch --export=ALL,THINK=1 ...  (or THINK=0)
_env_think = os.environ.get("THINK", "").strip().lower()
if _env_think:
    THINK = _env_think in ("1", "true", "yes", "on")

QWEN_MODEL = "qwen3.5:27b"
MODEL_TAG = "qwen_3_5_27b_fewshot"  # used in the output folder name
EMBED_MODEL = "qwen3-embedding:0.6b"

DATASETS_DIR = Path(
    "/lustre/nvwulf/scratch/ambasu/lexicon-framework"
    "/data_generation/event_keywords/generated_datasets"
)

# Separate output folders for thinking on / off, so they never overwrite.
PRED_DIR = Path(
    "/lustre/nvwulf/scratch/ambasu/lexicon-framework/baselines"
) / (f"{MODEL_TAG}_thinking_info_gap_predictions" if THINK
     else f"{MODEL_TAG}_info_gap_predictions")

OUT_FILE = PRED_DIR / "eval_info_gap_report.json"

# Ollama address comes from the SLURM job (unique port per job);
# falls back to the default port when run by hand.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_EMBED_URL = f"{OLLAMA_HOST}/api/embed"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

# --- Parallel run ---
# Number of conversation files processed at the same time.
# Must match OLLAMA_NUM_PARALLEL in the SLURM script.
NUM_WORKERS = 4

# Skip files that already have a complete prediction file in PRED_DIR
# (lets a stopped job continue where it ended).
RESUME = True

# --- Few-shot examples ---
# Worked examples (input -> expected output) are shown to the model before each
# real turn. They are taken from the ground truth of a few "demo" conversations,
# and those conversations are LEFT OUT of the evaluation, so nothing leaks.
# Leave FEWSHOT_FILES empty to pick N_FEWSHOT_FILES automatically (fixed seed,
# so every run and both tasks pick the same files), or list file names, e.g.
# FEWSHOT_FILES = ["client-dev_002.json", "client-dev_004.json"]
FEWSHOT_FILES = []
N_FEWSHOT_FILES = 2
N_POSITIVE_SHOTS = 4      # example turns that DO have a ground-truth answer
N_NEGATIVE_SHOTS = 2      # example turns with nothing to report (empty output)
FEWSHOT_HISTORY_TURNS = 6 # earlier turns shown in each example (keeps prompts short)
FEWSHOT_SEED = 42

FEWSHOT_MESSAGES = []     # filled in main()
FEWSHOT_INFO = {}         # recorded in the report

# Ground-truth key inside each dataset file (never shown to Qwen).
GT_KEY = "required_protocol_queries"

# ---- Same as the GPT-4.1 baseline ----
# How many top-level keys to send to LLM (after unwrapping)
NUM_KEYS_TO_SEND = 2

# If we can't reliably take first N keys, fall back to this preference order
FALLBACK_KEY_ORDER = ["backstory", "mobile_context_snapshot", "conversation_transcript", "dataset_id"]

# Optional: keep or drop LOW_VALUE entries (same flag as the GPT-4.1 baseline).
# True  -> LOW_VALUE is kept in GT and predictions and scored like any gap.
# False -> LOW_VALUE is dropped from both GT and predictions.
KEEP_LOW_VALUE = True
ACTIONABLE_VALUES = {"HIGH_VALUE", "MEDIUM_VALUE"}

# --- Matching a predicted gap to a GT gap (same turn) ---
# A pair is a candidate match if EITHER:
#   slot similarity      >= SLOT_MATCH_THRESHOLD   (same missing slot), OR
#   question similarity  >= QUESTION_MATCH_THRESHOLD (same clarifying question)
# slot similarity = max(lexical ratio, embedding cosine) on normalized slot names.
SLOT_MATCH_THRESHOLD = 0.75
QUESTION_MATCH_THRESHOLD = 0.85

# --- Correctness of a matched gap (analog of resolution >= 0.70) ---
# TP = matched gap whose natural_language_fallback similarity >= threshold.
RESOLUTION_THRESHOLD = 0.70

# If True, a matched gap is TP only when the intent also matches.
REQUIRE_INTENT_MATCH = False

TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42
# Ollama's default context window is small (2048-4096 tokens depending on
# version) and it silently cuts the START of the prompt when it overflows,
# which removes the system prompt. Prompt + NUM_PREDICT must fit in NUM_CTX.
if THINK:
    # Thinking tokens count toward num_predict. 4096 is too small with
    # thinking on: the model can run out before writing the JSON.
    NUM_PREDICT = 12288
    NUM_CTX = 32768
else:
    NUM_PREDICT = 4096   # same as MAX_TOKENS in the GPT-4.1 baseline
    NUM_CTX = 32768      # larger than zero-shot: the examples make the prompt longer
# Thinking turns can be long, especially with 4 requests sharing the GPU.
REQUEST_TIMEOUT_S = 1800
CHAT_ATTEMPTS = 2       # 1 retry after a timeout / server error
KEEP_ALIVE = "30m"

INTENTS = [
    "RESOLVE_MISSING_ENTITY",
    "REQUEST_MISSING_CONTEXT",
    "DISAMBIGUATE_REFERENCE",
    "REQUEST_PERMISSION",
    "REQUEST_TIMEFRAME",
    "REQUEST_LOCATION",
    "REQUEST_IDENTIFIER",
]
URGENCIES = ["IMMEDIATE", "ROUTINE", "NONE"]
VALUES = ["HIGH_VALUE", "MEDIUM_VALUE", "LOW_VALUE"]


# ============================================================
# HELPERS
# ============================================================

def safe_div(a, b):
    return float(a / b) if b else 0.0


def median_or_none(xs):
    return float(statistics.median(xs)) if xs else None


def mean_or_none(xs):
    return float(statistics.mean(xs)) if xs else None


def ns_to_s(x):
    return float(x or 0) / 1_000_000_000.0


def normalize_text(s):
    s = str(s or "").strip().lower()
    s = s.replace("*", "")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9'\s-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" -'\"")


def normalize_slot(s):
    """meetingTime / meeting_time / meeting-time -> 'meeting time'."""
    s = str(s or "")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[_\-./:]+", " ", s)
    return normalize_text(s)


def lexical_similarity(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return float(SequenceMatcher(None, a, b).ratio())


def coerce_enum(value, allowed, default):
    v = str(value or "").strip().upper()
    return v if v in allowed else default


def is_actionable(gap):
    return gap["query_quality_check"] in ACTIONABLE_VALUES


# ============================================================
# DATASET LOADING (from the GPT-4.1 script)
# ============================================================

def unwrap_top_level(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    raise ValueError(
        "Unsupported JSON shape: expected dict or list-of-dict."
    )


def normalize_transcript(obj):
    if isinstance(obj, dict) and isinstance(obj.get("conversation_transcript"), list):
        return _coerce_turn_list(obj["conversation_transcript"])

    for key in ("messages", "conversation", "turns"):
        if isinstance(obj, dict) and isinstance(obj.get(key), list):
            return _coerce_turn_list(obj[key])

    if isinstance(obj, list):
        return _coerce_turn_list(obj)

    return []


def _coerce_turn_list(turns):
    out = []
    for i, t in enumerate(turns, start=1):
        if not isinstance(t, dict):
            continue

        # Explicit None checks: a turn_id of 0 is valid and must not
        # fall through to the position index (it would misalign GT).
        turn_id = next(
            (t[k] for k in ("turn_id", "id", "turn") if t.get(k) is not None),
            i,
        )
        speaker = t.get("speaker") or t.get("role") or t.get("from") or "Unknown"
        text = t.get("text") or t.get("content") or t.get("message") or ""

        try:
            turn_id = int(turn_id)
        except Exception:
            turn_id = i

        out.append({"turn_id": turn_id, "speaker": str(speaker), "text": str(text)})
    return out


def transcript_until(transcript, turn_id):
    rows = [t for t in transcript if int(t["turn_id"]) <= int(turn_id)]
    rows.sort(key=lambda x: int(x["turn_id"]))
    return rows


def flatten_gap(row, turn_id=None):
    """
    Accepts the protocol shape:
      {trigger_turn_id, query_quality_check, reason,
       protocol_payload:{intent,target_slot,urgency,context_ref},
       natural_language_fallback}
    or a flat variant, and returns one flat dict.
    """
    payload = row.get("protocol_payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    def pick(key):
        return payload.get(key, row.get(key, ""))

    tid = row.get("trigger_turn_id", turn_id)
    try:
        tid = int(tid)
    except Exception:
        tid = turn_id

    return {
        "trigger_turn_id": tid,
        "query_quality_check": coerce_enum(
            row.get("query_quality_check"), VALUES, "MEDIUM_VALUE"
        ),
        "intent": coerce_enum(pick("intent"), INTENTS, "REQUEST_MISSING_CONTEXT"),
        "target_slot": str(pick("target_slot") or "").strip(),
        "urgency": coerce_enum(pick("urgency"), URGENCIES, "ROUTINE"),
        "context_ref": str(pick("context_ref") or "").strip(),
        "natural_language_fallback": str(
            row.get("natural_language_fallback") or ""
        ).strip(),
        "reason": str(row.get("reason") or "").strip(),
    }


def to_protocol_shape(gap):
    """Flat gap -> the same output shape the GPT-4.1 baseline writes."""
    return {
        "trigger_turn_id": gap["trigger_turn_id"],
        "query_quality_check": gap["query_quality_check"],
        "reason": gap.get("reason", ""),
        "protocol_payload": {
            "intent": gap["intent"],
            "target_slot": gap["target_slot"],
            "urgency": gap["urgency"],
            "context_ref": gap["context_ref"],
        },
        "natural_language_fallback": gap["natural_language_fallback"],
    }


def group_gt_by_turn(gt_rows):
    """Returns (grouped_by_turn, number_of_unusable_rows)."""
    grouped = defaultdict(list)
    unusable = 0
    for gt_id, row in enumerate(gt_rows):
        if not isinstance(row, dict):
            unusable += 1
            continue
        item = flatten_gap(row)
        if item["trigger_turn_id"] is None:
            unusable += 1
            continue
        item["_gt_id"] = gt_id
        grouped[int(item["trigger_turn_id"])].append(item)
    return grouped, unusable


# ============================================================
# OLLAMA
# ============================================================

def check_ollama(session):
    r = session.get(OLLAMA_TAGS_URL, timeout=15)
    r.raise_for_status()

    names = {m.get("name") for m in r.json().get("models", []) if m.get("name")}
    missing = [m for m in [QWEN_MODEL, EMBED_MODEL] if m not in names]

    if missing:
        raise RuntimeError(
            "Missing Ollama model(s): "
            + ", ".join(missing)
            + "\nRun:\n"
            + "\n".join(f"ollama pull {m}" for m in missing)
        )


# =========================
# SYSTEM PROMPT (same as the GPT-4.1 baseline)
# =========================
SYSTEM_INSTRUCTIONS = """
You are an expert at detecting information gaps in multi-party dialogues.

CRITICAL:
- Output ONLY valid JSON (no markdown, no commentary).
- Your output MUST be a JSON object with TWO keys:
  1) "required_protocol_queries"
  2) "generation_metadata"

required_protocol_queries:
- MUST be a list of objects, each exactly of the form:

{
  "trigger_turn_id": int,
  "query_quality_check": "HIGH_VALUE" | "MEDIUM_VALUE" | "LOW_VALUE",
  "reason": "",
  "protocol_payload": {
    "intent": "RESOLVE_MISSING_ENTITY" | "REQUEST_MISSING_CONTEXT" | "DISAMBIGUATE_REFERENCE" | "REQUEST_PERMISSION" | "REQUEST_TIMEFRAME" | "REQUEST_LOCATION" | "REQUEST_IDENTIFIER",
    "target_slot": "",
    "urgency": "IMMEDIATE" | "ROUTINE" | "NONE",
    "context_ref": ""
  },
  "natural_language_fallback": ""
}

GOAL:
- Find turns where an assistant would be blocked, or likely to make an incorrect assumption, due to missing information.

RULES:
- Prefer HIGH precision for HIGH/MEDIUM_VALUE.
- LOW_VALUE items are allowed ONLY when clearly non-actionable (small talk / already-known / not needed).
- "reason" must be 1–2 sentences.
- "natural_language_fallback" must be a single question/message that would close the gap (or a short note if LOW_VALUE says it should NOT trigger).
- If there are no meaningful gaps, return an empty list.

generation_metadata:
- MUST be exactly:

{
  "model": "",
  "temperature": number,
  "generated_at": "",
  "scenario_request": "",
  "provider": "",
  "endpoint": ""
}

- Copy these fields from the input metadata provided by the user message.
""".strip()


# =========================
# Strict JSON schema (same as the GPT-4.1 baseline)
# Ollama takes the inner JSON schema directly in "format".
# =========================
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "required_protocol_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "trigger_turn_id": {"type": "integer"},
                    "query_quality_check": {"type": "string", "enum": VALUES},
                    "reason": {"type": "string"},
                    "protocol_payload": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "intent": {"type": "string", "enum": INTENTS},
                            "target_slot": {"type": "string"},
                            "urgency": {"type": "string", "enum": URGENCIES},
                            "context_ref": {"type": "string"},
                        },
                        "required": ["intent", "target_slot", "urgency", "context_ref"],
                    },
                    "natural_language_fallback": {"type": "string"},
                },
                "required": [
                    "trigger_turn_id",
                    "query_quality_check",
                    "reason",
                    "protocol_payload",
                    "natural_language_fallback",
                ],
            },
        },
        "generation_metadata": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model": {"type": "string"},
                "temperature": {"type": "number"},
                "generated_at": {"type": "string"},
                "scenario_request": {"type": "string"},
                "provider": {"type": "string"},
                "endpoint": {"type": "string"},
            },
            "required": [
                "model",
                "temperature",
                "generated_at",
                "scenario_request",
                "provider",
                "endpoint",
            ],
        },
    },
    "required": ["required_protocol_queries", "generation_metadata"],
}


# =========================
# Build "only first N keys" payload (same as the GPT-4.1 baseline)
# =========================
def pick_first_n_keys(d, n):
    keys = list(d.keys())
    if len(keys) >= n:
        return keys[:n]
    return keys


def build_llm_payload_only_first_keys(top_obj, transcript_norm, n_keys=2):
    """
    Build payload that contains ONLY the first n_keys top-level keys (in file order),
    BUT:
      - if 'conversation_transcript' is included, replace it with normalized transcript
      - never include ground_truth_resolutions / required_protocol_queries (avoid leakage)
    If first n keys are unusable (e.g. missing transcript), fallback to preferred keys.
    """
    blocked_keys = {"ground_truth_resolutions", "required_protocol_queries", "generation_metadata"}
    first_keys = [k for k in pick_first_n_keys(top_obj, n_keys) if k not in blocked_keys]

    payload = {}
    for k in first_keys:
        if k == "conversation_transcript":
            payload[k] = transcript_norm
        else:
            payload[k] = top_obj.get(k)

    # Ensure transcript is present if we have it
    if "conversation_transcript" not in payload and transcript_norm:
        if len(payload) >= n_keys:
            last_key = list(payload.keys())[-1]
            payload.pop(last_key, None)
        payload["conversation_transcript"] = transcript_norm

    # Fallback order if needed
    if not payload:
        fallback = []
        for k in FALLBACK_KEY_ORDER:
            if k in top_obj and k not in blocked_keys:
                fallback.append(k)
            if len(fallback) == n_keys:
                break

        payload = {}
        for k in fallback:
            if k == "conversation_transcript":
                payload[k] = transcript_norm
            else:
                payload[k] = top_obj.get(k)

    # Replace None with ""
    for k in list(payload.keys()):
        if payload[k] is None:
            payload[k] = ""

    return payload


def make_prompt(top_obj, transcript, current_turn, metadata,
                history_turns=None, with_examples=True):
    """
    Same user_task wrapper as the GPT-4.1 baseline, run per turn:
    the transcript inside the payload is cut at the current turn,
    so Qwen never sees future turns.
    """
    turn_id = int(current_turn["turn_id"])

    llm_payload = build_llm_payload_only_first_keys(
        top_obj=top_obj,
        transcript_norm=transcript_until(transcript, turn_id)[-history_turns:]
        if history_turns else transcript_until(transcript, turn_id),
        n_keys=NUM_KEYS_TO_SEND,
    )

    user_task = {
        "task": "Detect information gaps per user turn and propose required protocol queries.",
        "instructions": [
            "Scan the conversation turns and identify missing information that blocks an assistant.",
            "Create protocol queries only when truly necessary to proceed accurately.",
            "HIGH/MEDIUM_VALUE must be actionable gaps.",
            "LOW_VALUE is allowed only for clearly non-actionable mentions; set urgency to NONE for LOW_VALUE.",
            "Return JSON with keys required_protocol_queries and generation_metadata.",
            "generation_metadata must mirror the input metadata fields exactly.",
            # Only addition for per-turn evaluation:
            f"Evaluate ONLY turn {turn_id} (current_turn_id); every trigger_turn_id must be {turn_id}.",
        ],
        "current_turn_id": turn_id,
        "input": llm_payload,
        "metadata": metadata,
    }

    messages = (
        [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
        + (FEWSHOT_MESSAGES if with_examples else [])
        + [{"role": "user", "content": json.dumps(user_task, ensure_ascii=False)}]
    )
    return messages, list(llm_payload.keys())



def pick_fewshot_files(files):
    """Choose the demo conversations (same choice for both tasks)."""
    if FEWSHOT_FILES:
        by_name = {p.name: p for p in files}
        missing = [n for n in FEWSHOT_FILES if n not in by_name]
        if missing:
            raise FileNotFoundError(f"FEWSHOT_FILES not found in datasets: {missing}")
        return [by_name[n] for n in FEWSHOT_FILES]
    rng = random.Random(FEWSHOT_SEED)
    return sorted(rng.sample(sorted(files), min(N_FEWSHOT_FILES, len(files) - 1)))


def pick_shots(positives, negatives, diversity_key):
    """Deterministic pick: positives spread over different answer types, then negatives."""
    rng = random.Random(FEWSHOT_SEED + 1)
    positives = list(positives)
    negatives = list(negatives)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    chosen, seen = [], set()
    for item in positives:                      # first pass: one per answer type
        k = diversity_key(item)
        if k not in seen and len(chosen) < N_POSITIVE_SHOTS:
            chosen.append(item)
            seen.add(k)
    for item in positives:                      # fill up if needed
        if len(chosen) >= N_POSITIVE_SHOTS:
            break
        if item not in chosen:
            chosen.append(item)
    chosen += negatives[:N_NEGATIVE_SHOTS]

    if len(chosen) < N_POSITIVE_SHOTS + N_NEGATIVE_SHOTS:
        print(f"  WARNING: only {len(chosen)} few-shot examples available")
    rng.shuffle(chosen)                         # mix positive / negative order
    return chosen


def build_fewshot_messages(demo_files):
    """Example user/assistant message pairs from the demo conversations' ground truth."""
    positives, negatives = [], []
    for path in demo_files:
        with path.open("r", encoding="utf-8") as f:
            data = unwrap_top_level(json.load(f))
        transcript = normalize_transcript(data)
        gt_map, _ = group_gt_by_turn(data.get(GT_KEY) or [])
        scenario_request = (
            data.get("scenario_request")
            or data.get("backstory")
            or data.get("mobile_context_snapshot")
            or ""
        )
        metadata = build_metadata(scenario_request)
        for turn in transcript:
            tid = int(turn["turn_id"])
            item = (path.name, tid, data, transcript, turn, metadata, gt_map.get(tid, []))
            (positives if item[6] else negatives).append(item)

    chosen = pick_shots(positives, negatives, diversity_key=lambda it: it[6][0]["intent"])

    messages, info = [], []
    for name, tid, data, transcript, turn, metadata, gts in chosen:
        user_msgs, _ = make_prompt(data, transcript, turn, metadata,
                                   history_turns=FEWSHOT_HISTORY_TURNS, with_examples=False)
        answer = {
            "required_protocol_queries": [to_protocol_shape(g) for g in gts],
            "generation_metadata": metadata,
        }
        messages.append(user_msgs[-1])
        messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
        info.append({"file": name, "turn_id": tid, "n_gaps": len(gts)})
    return messages, info


def build_metadata(scenario_request):
    return {
        "model": QWEN_MODEL,
        "temperature": float(TEMPERATURE),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_request": str(scenario_request),
        "provider": "Ollama",
        "endpoint": OLLAMA_CHAT_URL,
    }



def post_chat_with_retry(session, payload):
    """POST to Ollama chat. Retries timeouts, dropped connections and server
    errors (5xx). Returns (response, None) on success or repeat-limit, or
    (None, error_text) if every attempt failed - the caller then counts that
    turn as "no prediction" instead of failing the whole conversation."""
    last_error = ""
    for attempt in range(1, CHAT_ATTEMPTS + 1):
        try:
            r = session.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT_S)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = repr(exc)
        else:
            if r.ok or "repeat limit" in r.text:
                return r, None
            if r.status_code < 500:
                # A real request problem (e.g. model not found): stop, as before.
                raise RuntimeError(f"Ollama chat failed ({r.status_code}): {r.text}")
            last_error = f"HTTP {r.status_code}: {r.text[:300]}"
        if attempt < CHAT_ATTEMPTS:
            print(f"  retrying turn (attempt {attempt + 1}/{CHAT_ATTEMPTS}) after: {last_error[:200]}")
            time.sleep(15 * attempt)
    print(f"  giving up on this turn after {CHAT_ATTEMPTS} attempts: {last_error[:200]}")
    return None, last_error


def post_embed_with_retry(session, url, **kwargs):
    """POST to Ollama embed, retrying timeouts / dropped connections / 5xx."""
    last_error = ""
    for attempt in range(1, 4):
        try:
            r = session.post(url, **kwargs)
            if r.ok or r.status_code < 500:
                return r
            last_error = f"HTTP {r.status_code}: {r.text[:300]}"
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = repr(exc)
        if attempt < 3:
            time.sleep(10 * attempt)
    raise RuntimeError(f"Ollama embed failed after 3 attempts: {last_error}")


def call_qwen(session, messages, turn_id):
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "think": THINK,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }

    t0 = time.perf_counter()
    r, request_error = post_chat_with_retry(session, payload)
    latency = time.perf_counter() - t0

    repeat_aborted = False
    request_failed = False
    if r is None:
        # Timed out / server error on every attempt: count this turn as
        # "no prediction" instead of failing the whole conversation.
        request_failed = True
        raw = {"message": {"content": ""}}
    elif not r.ok:
        # Model got stuck repeating itself and Ollama aborted it.
        repeat_aborted = True
        raw = {"message": {"content": ""}}
    else:
        raw = r.json()

    content = raw.get("message", {}).get("content", "")
    thinking = raw.get("message", {}).get("thinking", "") or ""

    # A truncated generation can break JSON:
    # count it as "no queries" for this turn instead of failing the file.
    parse_ok = True
    try:
        parsed = json.loads(content)
        queries = parsed.get("required_protocol_queries", []) if isinstance(parsed, dict) else []
        if not isinstance(queries, list):
            queries = []
    except json.JSONDecodeError:
        parse_ok = False
        queries = []

    cleaned = []
    for q in queries:
        if not isinstance(q, dict):
            continue
        flat = flatten_gap(q, turn_id=turn_id)
        flat["trigger_turn_id"] = turn_id
        if flat["target_slot"] or flat["natural_language_fallback"]:
            cleaned.append(flat)

    prompt_eval_s = ns_to_s(raw.get("prompt_eval_duration"))
    generation_s = ns_to_s(raw.get("eval_duration"))
    prompt_tokens = int(raw.get("prompt_eval_count") or 0)
    output_tokens = int(raw.get("eval_count") or 0)

    compute = {
        "wall_latency_s": latency,
        "ollama_total_s": ns_to_s(raw.get("total_duration")),
        "model_load_s": ns_to_s(raw.get("load_duration")),
        "prompt_eval_s": prompt_eval_s,
        "generation_s": generation_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "generation_tokens_per_s": safe_div(output_tokens, generation_s),
        "json_parse_ok": parse_ok,
        "thinking_returned": bool(thinking.strip()),
        "thinking_chars": len(thinking),
        "hit_num_predict_limit": output_tokens >= NUM_PREDICT,
        "repeat_aborted": repeat_aborted,
        "request_failed": request_failed,
        "request_error": request_error,
        "context_overflow_risk": prompt_tokens + NUM_PREDICT > NUM_CTX,
    }

    if compute["context_overflow_risk"]:
        print(
            f"  WARNING: turn {turn_id}: prompt ({prompt_tokens} tok) + "
            f"NUM_PREDICT ({NUM_PREDICT}) > NUM_CTX ({NUM_CTX}); raise NUM_CTX."
        )

    return cleaned, compute


def warm_up(session):
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": "Return JSON with an empty required_protocol_queries list."}],
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "think": THINK,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 128, "num_ctx": NUM_CTX},
    }

    r = session.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT_S)
    if not r.ok:
        raise RuntimeError(f"Qwen warm-up failed ({r.status_code}): {r.text}")


# ============================================================
# EMBEDDINGS (cached, unit-normalized)
# ============================================================

_EMBED_CACHE = {}


def embed_texts(session, texts):
    texts = [str(t or "") for t in texts]
    to_call = sorted({t for t in texts if t not in _EMBED_CACHE and t.strip()})

    if to_call:
        r = post_embed_with_retry(
            session,
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "input": to_call, "keep_alive": KEEP_ALIVE},
            timeout=REQUEST_TIMEOUT_S,
        )
        if not r.ok:
            raise RuntimeError(f"Ollama embed failed ({r.status_code}): {r.text}")

        embs = r.json().get("embeddings", [])
        if len(embs) != len(to_call):
            raise ValueError(f"Expected {len(to_call)} embeddings, got {len(embs)}")

        for t, e in zip(to_call, embs):
            v = np.asarray(e, dtype=np.float64)
            n = np.linalg.norm(v)
            _EMBED_CACHE[t] = v / n if n else None

    return [_EMBED_CACHE.get(t) if t.strip() else None for t in texts]


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b))))


# ============================================================
# GAP MATCHING
# ============================================================

def similarity_matrices(session, predictions, gt_items):
    P, G = len(predictions), len(gt_items)
    slot_sim = np.zeros((P, G))
    q_sim = np.zeros((P, G))

    if not P or not G:
        return slot_sim, q_sim

    p_slots = [normalize_slot(p["target_slot"]) for p in predictions]
    g_slots = [normalize_slot(g["target_slot"]) for g in gt_items]
    p_qs = [p["natural_language_fallback"] for p in predictions]
    g_qs = [g["natural_language_fallback"] for g in gt_items]

    embs = embed_texts(session, p_slots + g_slots + p_qs + g_qs)
    e_ps = embs[:P]
    e_gs = embs[P:P + G]
    e_pq = embs[P + G:2 * P + G]
    e_gq = embs[2 * P + G:]

    for i in range(P):
        for j in range(G):
            slot_sim[i, j] = max(
                lexical_similarity(p_slots[i], g_slots[j]),
                cosine(e_ps[i], e_gs[j]),
            )
            q_sim[i, j] = cosine(e_pq[i], e_gq[j])

    return slot_sim, q_sim


def match_gaps(slot_sim, q_sim):
    """
    Greedy one-to-one matching of predicted gaps to GT gaps within a turn.

    Returns:
      matches = [(pred_index, gt_index, basis), ...]
      unmatched_pred_indices
      unmatched_gt_indices
    """
    P, G = slot_sim.shape
    candidates = []

    for i in range(P):
        for j in range(G):
            slot_ok = slot_sim[i, j] >= SLOT_MATCH_THRESHOLD
            q_ok = q_sim[i, j] >= QUESTION_MATCH_THRESHOLD
            if slot_ok or q_ok:
                basis = "slot+question" if (slot_ok and q_ok) else ("slot" if slot_ok else "question")
                candidates.append((slot_sim[i, j] + q_sim[i, j], i, j, basis))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_p, used_g, matches = set(), set(), []
    for _, i, j, basis in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, basis))

    unmatched_p = [i for i in range(P) if i not in used_p]
    unmatched_g = [j for j in range(G) if j not in used_g]
    return matches, unmatched_p, unmatched_g


# ============================================================
# TURN EVALUATION
# ============================================================

def evaluate_turn(session, current_turn, predictions_all, gt_all, compute):
    if KEEP_LOW_VALUE:
        predictions = list(predictions_all)
        gt_items = list(gt_all)
    else:
        predictions = [p for p in predictions_all if is_actionable(p)]
        gt_items = [g for g in gt_all if is_actionable(g)]

    slot_sim, q_sim = similarity_matrices(session, predictions, gt_items)
    matches, unmatched_p, unmatched_g = match_gaps(slot_sim, q_sim)

    matched_rows = []
    tp = 0
    bad_resolution = 0

    for pi, gi, basis in matches:
        pred = predictions[pi]
        gt = gt_items[gi]

        q_score = float(q_sim[pi, gi])
        intent_match = pred["intent"] == gt["intent"]

        is_tp = q_score >= RESOLUTION_THRESHOLD
        if REQUIRE_INTENT_MATCH:
            is_tp = is_tp and intent_match

        if is_tp:
            tp += 1
        else:
            bad_resolution += 1

        matched_rows.append({
            "match_basis": basis,
            "slot_similarity": float(slot_sim[pi, gi]),
            "question_similarity": q_score,
            "is_true_positive": is_tp,

            "predicted_target_slot": pred["target_slot"],
            "ground_truth_target_slot": gt["target_slot"],
            "predicted_question": pred["natural_language_fallback"],
            "ground_truth_question": gt["natural_language_fallback"],
            "predicted_context_ref": pred["context_ref"],
            "ground_truth_context_ref": gt["context_ref"],

            "predicted_intent": pred["intent"],
            "ground_truth_intent": gt["intent"],
            "intent_match": intent_match,

            "predicted_urgency": pred["urgency"],
            "ground_truth_urgency": gt["urgency"],
            "urgency_match": pred["urgency"] == gt["urgency"],

            "predicted_value": pred["query_quality_check"],
            "ground_truth_value": gt["query_quality_check"],
            "value_match": pred["query_quality_check"] == gt["query_quality_check"],
        })

    false_positives = [to_protocol_shape(predictions[i]) for i in unmatched_p]

    missed_gt = [
        {
            "target_slot": gt_items[i]["target_slot"],
            "intent": gt_items[i]["intent"],
            "ground_truth_value": gt_items[i]["query_quality_check"],
            "context_ref": gt_items[i]["context_ref"],
            "ground_truth_question": gt_items[i]["natural_language_fallback"],
        }
        for i in unmatched_g
    ]

    # Gap-level definitions:
    #   TP = matched GT gap + question similarity >= RESOLUTION_THRESHOLD
    #   FN = missed GT gap OR matched GT gap below threshold
    #   FP = predicted gap that matched no GT gap
    fn = len(unmatched_g) + bad_resolution
    fp = len(unmatched_p)

    return {
        "turn_id": int(current_turn["turn_id"]),
        "speaker": current_turn.get("speaker"),
        "text": current_turn.get("text", ""),

        "counts": {
            "ground_truth_gaps": len(gt_items),
            "predicted_gaps": len(predictions),
            "detected_gt_gaps": len(matches),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "missed_gt_gaps": len(unmatched_g),
            "detected_but_question_below_threshold": bad_resolution,
            "excluded_low_value_gt": len(gt_all) - len(gt_items),
            "excluded_low_value_predictions": len(predictions_all) - len(predictions),
        },

        "turn_has_gt_gap": bool(gt_items),
        "turn_has_predicted_gap": bool(predictions),

        "matched": matched_rows,
        "false_positive_predictions": false_positives,
        "missed_ground_truth": missed_gt,
        "required_protocol_queries": [to_protocol_shape(p) for p in predictions_all],
        "compute": compute,
    }


# ============================================================
# METRICS
# ============================================================

def metrics_from_turns(turns):
    def total(key):
        return sum(t["counts"][key] for t in turns)

    gt = total("ground_truth_gaps")
    pred = total("predicted_gaps")
    detected = total("detected_gt_gaps")
    tp = total("true_positives")
    fn = total("false_negatives")
    fp = total("false_positives")
    missed = total("missed_gt_gaps")
    bad_res = total("detected_but_question_below_threshold")

    # --- Gap level: detection only ---
    detection_recall = safe_div(detected, gt)
    detection_precision = safe_div(detected, detected + fp)
    detection_f1 = safe_div(
        2 * detection_precision * detection_recall,
        detection_precision + detection_recall,
    )

    # --- Gap level: detection + correct clarifying question ---
    tpr = safe_div(tp, gt)
    strict_precision = safe_div(tp, pred)
    end_to_end_f1 = safe_div(2 * strict_precision * tpr, strict_precision + tpr)

    # --- Turn level: does the turn have any (kept) gap? ---
    # Every turn is a finite candidate, so classical FPR is defined here.
    t_tp = sum(1 for t in turns if t["turn_has_gt_gap"] and t["turn_has_predicted_gap"])
    t_fp = sum(1 for t in turns if not t["turn_has_gt_gap"] and t["turn_has_predicted_gap"])
    t_fn = sum(1 for t in turns if t["turn_has_gt_gap"] and not t["turn_has_predicted_gap"])
    t_tn = sum(1 for t in turns if not t["turn_has_gt_gap"] and not t["turn_has_predicted_gap"])

    t_precision = safe_div(t_tp, t_tp + t_fp)
    t_recall = safe_div(t_tp, t_tp + t_fn)

    # --- Attribute agreement on matched pairs ---
    matched = [m for t in turns for m in t["matched"]]
    missed_rows = [m for t in turns for m in t["missed_ground_truth"]]

    # --- Recall per GT value class ---
    by_value = {}
    for v in VALUES:
        v_matched = [m for m in matched if m["ground_truth_value"] == v]
        v_missed = [m for m in missed_rows if m["ground_truth_value"] == v]
        v_gt = len(v_matched) + len(v_missed)
        if v_gt == 0:
            continue
        v_tp = sum(1 for m in v_matched if m["is_true_positive"])
        by_value[v] = {
            "ground_truth_gaps": v_gt,
            "detected": len(v_matched),
            "true_positives": v_tp,
            "detection_recall": safe_div(len(v_matched), v_gt),
            "true_positive_rate": safe_div(v_tp, v_gt),
        }

    return {
        "ground_truth_gaps": gt,
        "qwen_predicted_gaps": pred,
        "detected_ground_truth_gaps": detected,
        "missed_ground_truth_gaps": missed,
        "detected_but_bad_question": bad_res,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positive_gaps": fp,

        "detection_recall": detection_recall,
        "detection_precision": detection_precision,
        "detection_f1": detection_f1,

        "true_positive_rate_at_threshold": tpr,
        "strict_end_to_end_precision_at_threshold": strict_precision,
        "end_to_end_f1_at_threshold": end_to_end_f1,

        "false_positive_discovery_rate": safe_div(fp, pred),
        "false_positives_per_turn": safe_div(fp, len(turns)),

        "turn_level": {
            "turns": len(turns),
            "tp": t_tp,
            "fp": t_fp,
            "fn": t_fn,
            "tn": t_tn,
            "precision": t_precision,
            "recall": t_recall,
            "f1": safe_div(2 * t_precision * t_recall, t_precision + t_recall),
            "false_positive_rate": safe_div(t_fp, t_fp + t_tn),
            "specificity": safe_div(t_tn, t_fp + t_tn),
            "accuracy": safe_div(t_tp + t_tn, len(turns)),
        },

        "matched_attribute_agreement": {
            "matched_pairs": len(matched),
            "intent_accuracy": safe_div(sum(m["intent_match"] for m in matched), len(matched)),
            "urgency_accuracy": safe_div(sum(m["urgency_match"] for m in matched), len(matched)),
            "value_accuracy": safe_div(sum(m["value_match"] for m in matched), len(matched)),
            "mean_slot_similarity": mean_or_none([m["slot_similarity"] for m in matched]),
            "mean_question_similarity": mean_or_none([m["question_similarity"] for m in matched]),
        },

        "by_ground_truth_value": by_value,

        "excluded_low_value_gt": total("excluded_low_value_gt"),
        "excluded_low_value_predictions": total("excluded_low_value_predictions"),
    }


def latency_from_turns(turns):
    latencies = [t["compute"]["wall_latency_s"] for t in turns]

    return {
        "number_of_turn_calls": len(turns),
        "median_latency_per_turn_s": median_or_none(latencies),
        "mean_latency_per_turn_s": mean_or_none(latencies),
        "total_prompt_tokens": sum(t["compute"]["prompt_tokens"] for t in turns),
        "total_output_tokens": sum(t["compute"]["output_tokens"] for t in turns),
        "json_parse_failures": sum(1 for t in turns if not t["compute"]["json_parse_ok"]),
        "context_overflow_risk_turns": sum(
            1 for t in turns if t["compute"].get("context_overflow_risk")
        ),
        "turns_with_thinking": sum(
            1 for t in turns if t["compute"].get("thinking_returned")
        ),
        "turns_hit_num_predict_limit": sum(
            1 for t in turns if t["compute"].get("hit_num_predict_limit")
        ),
        "turns_repeat_aborted": sum(
            1 for t in turns if t["compute"].get("repeat_aborted")
        ),
        "turns_request_failed": sum(
            1 for t in turns if t["compute"].get("request_failed")
        ),
    }


# ============================================================
# ONE FILE
# ============================================================

def process_file(path, session):
    with path.open("r", encoding="utf-8") as f:
        data = unwrap_top_level(json.load(f))

    transcript = normalize_transcript(data)
    if not transcript:
        raise ValueError("Could not find/normalize conversation transcript.")

    if GT_KEY not in data or not isinstance(data[GT_KEY], list):
        raise ValueError(f"Missing ground-truth list '{GT_KEY}'.")

    # Same as the GPT-4.1 baseline
    scenario_request = (
        data.get("scenario_request")
        or data.get("backstory")
        or data.get("mobile_context_snapshot")
        or ""
    )
    generation_metadata = build_metadata(scenario_request)
    llm_input_keys_sent = []

    gt_map, unusable_gt = group_gt_by_turn(data[GT_KEY])
    if unusable_gt:
        print(f"  WARNING: {unusable_gt} GT row(s) have no usable trigger_turn_id; skipped.")

    # GT items pointing at turns that do not exist would silently vanish.
    transcript_ids = {int(t["turn_id"]) for t in transcript}
    orphan_gt = sum(len(v) for k, v in gt_map.items() if k not in transcript_ids)
    if orphan_gt:
        print(f"  WARNING: {orphan_gt} GT gap(s) reference turns not in the transcript.")

    evaluated_turns = []
    seen_turn_ids = set()

    # Run Qwen on EVERY turn, so false positives on
    # turns with no GT gap are measured.
    for i, turn in enumerate(transcript, start=1):
        turn_id = int(turn["turn_id"])

        messages, llm_input_keys_sent = make_prompt(
            data, transcript, turn, generation_metadata
        )
        predictions, compute = call_qwen(session, messages, turn_id)

        evaluated = evaluate_turn(
            session=session,
            current_turn=turn,
            predictions_all=predictions,
            # GT is attached only to the first turn with this id,
            # so a duplicated turn_id cannot double-count GT gaps.
            gt_all=[] if turn_id in seen_turn_ids else gt_map.get(turn_id, []),
            compute=compute,
        )
        evaluated_turns.append(evaluated)
        seen_turn_ids.add(turn_id)

        c = evaluated["counts"]
        print(
            f"  {path.stem} turn {turn_id:>3} ({i}/{len(transcript)}) "
            f"GT={c['ground_truth_gaps']} "
            f"PRED={c['predicted_gaps']} "
            f"TP={c['true_positives']} "
            f"FN={c['false_negatives']} "
            f"FP={c['false_positives']} "
            f"lat={compute['wall_latency_s']:.2f}s"
            + ("" if compute["json_parse_ok"] else " [JSON PARSE FAIL]")
            + (" [REPEAT ABORT]" if compute.get("repeat_aborted") else "")
            + (" [REQUEST FAILED]" if compute.get("request_failed") else "")
        )

    metrics = metrics_from_turns(evaluated_turns)
    metrics["orphan_ground_truth_gaps"] = orphan_gt
    metrics["unusable_ground_truth_rows"] = unusable_gt

    return {
        "dataset_id": data.get("dataset_id", ""),
        "file": path.name,
        "source_file": str(path),
        "llm_input_keys_sent": llm_input_keys_sent,
        "generation_metadata": generation_metadata,
        "metrics": metrics,
        "latency_compute": latency_from_turns(evaluated_turns),
        "required_protocol_queries": [
            q for t in evaluated_turns for q in t["required_protocol_queries"]
        ],
        "turns": evaluated_turns,
    }


# ============================================================
# GLOBAL REPORT
# ============================================================

def aggregate(file_results):
    all_turns = [turn for result in file_results for turn in result["turns"]]

    per_file = []
    for result in file_results:
        per_file.append({
            "dataset_id": result["dataset_id"],
            "source_file": result["source_file"],
            **result["metrics"],
            **result["latency_compute"],
        })

    overall_latency = latency_from_turns(all_turns)

    file_medians = [
        x["median_latency_per_turn_s"]
        for x in per_file
        if x["median_latency_per_turn_s"] is not None
    ]
    overall_latency["median_of_conversation_median_latencies_s"] = median_or_none(file_medians)

    overall = metrics_from_turns(all_turns)
    overall["orphan_ground_truth_gaps"] = sum(
        r["metrics"]["orphan_ground_truth_gaps"] for r in file_results
    )
    overall["unusable_ground_truth_rows"] = sum(
        r["metrics"]["unusable_ground_truth_rows"] for r in file_results
    )

    return {
        "configuration": {
            "qwen_model": QWEN_MODEL,
            "embedding_model": EMBED_MODEL,
            "ground_truth_key": GT_KEY,
            "think": THINK,
            "few_shot": FEWSHOT_INFO,
            "num_keys_to_send": NUM_KEYS_TO_SEND,
            "fallback_key_order": FALLBACK_KEY_ORDER,
            "keep_low_value": KEEP_LOW_VALUE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "slot_match_threshold": SLOT_MATCH_THRESHOLD,
            "question_match_threshold": QUESTION_MATCH_THRESHOLD,
            "resolution_similarity_threshold": RESOLUTION_THRESHOLD,
            "require_intent_match": REQUIRE_INTENT_MATCH,
            "ground_truth_shown_to_qwen": False,
        },
        "overall": overall,
        "latency_compute": overall_latency,
        "per_file": per_file,
    }


def print_summary(report):
    m = report["overall"]
    l = report["latency_compute"]
    tl = m["turn_level"]
    ag = m["matched_attribute_agreement"]
    thr = RESOLUTION_THRESHOLD

    print("\n" + "=" * 80)
    print("INFORMATION GAP DETECTION")
    print("=" * 80)

    print(f"GT gaps:                          {m['ground_truth_gaps']}")
    print(f"Qwen predicted gaps:              {m['qwen_predicted_gaps']}")
    print(f"Detected GT gaps:                 {m['detected_ground_truth_gaps']}")
    print(f"Missed GT gaps:                   {m['missed_ground_truth_gaps']}")
    print(f"Detected but question sim < {thr:.2f}: {m['detected_but_bad_question']}")
    print(f"TRUE POSITIVES:                   {m['true_positives']}")
    print(f"FALSE NEGATIVES:                  {m['false_negatives']}")
    print(f"FALSE POSITIVES:                  {m['false_positive_gaps']}")
    if not KEEP_LOW_VALUE:
        print(f"Excluded LOW_VALUE (GT / pred):   "
              f"{m['excluded_low_value_gt']} / {m['excluded_low_value_predictions']}")

    print("\nGap detection only:")
    print(f"  Recall:                         {100*m['detection_recall']:.2f}%")
    print(f"  Precision:                      {100*m['detection_precision']:.2f}%")
    print(f"  F1:                             {100*m['detection_f1']:.2f}%")

    print(f"\nGap detection + correct question (@ {thr:.2f}):")
    print(f"  TPR / Recall:                   {100*m['true_positive_rate_at_threshold']:.2f}%")
    print(f"  Precision:                      {100*m['strict_end_to_end_precision_at_threshold']:.2f}%")
    print(f"  F1:                             {100*m['end_to_end_f1_at_threshold']:.2f}%")

    print("\nTurn level (turn has any gap"
          + (", LOW_VALUE included):" if KEEP_LOW_VALUE else ", HIGH/MEDIUM only):"))
    print(f"  TP/FP/FN/TN:                    {tl['tp']}/{tl['fp']}/{tl['fn']}/{tl['tn']}")
    print(f"  Precision:                      {100*tl['precision']:.2f}%")
    print(f"  Recall:                         {100*tl['recall']:.2f}%")
    print(f"  F1:                             {100*tl['f1']:.2f}%")
    print(f"  FPR:                            {100*tl['false_positive_rate']:.2f}%")
    print(f"  Accuracy:                       {100*tl['accuracy']:.2f}%")

    print("\nMatched-pair agreement:")
    print(f"  Intent accuracy:                {100*ag['intent_accuracy']:.2f}%")
    print(f"  Urgency accuracy:               {100*ag['urgency_accuracy']:.2f}%")
    print(f"  Value accuracy:                 {100*ag['value_accuracy']:.2f}%")

    if m["by_ground_truth_value"]:
        print("\nRecall by GT value:")
        for v, row in m["by_ground_truth_value"].items():
            print(f"  {v:<14} n={row['ground_truth_gaps']:<5} "
                  f"det={100*row['detection_recall']:.2f}%  "
                  f"TPR={100*row['true_positive_rate']:.2f}%")

    print("\nFalse positives:")
    print(f"  FP discovery rate:              {100*m['false_positive_discovery_rate']:.2f}%")
    print(f"  FP per turn:                    {m['false_positives_per_turn']:.4f}")

    print("\nLatency:")
    if l["median_latency_per_turn_s"] is not None:
        print(f"  Median turn latency:            {l['median_latency_per_turn_s']:.3f}s")
    if l["median_of_conversation_median_latencies_s"] is not None:
        print(f"  Median of file medians:         {l['median_of_conversation_median_latencies_s']:.3f}s")
    print(f"  JSON parse failures:            {l['json_parse_failures']}")
    print(f"  Turns with thinking:            {l.get('turns_with_thinking', 0)}")
    print(f"  Turns hit NUM_PREDICT limit:    {l.get('turns_hit_num_predict_limit', 0)}")
    print(f"  Turns aborted (repeat limit):   {l.get('turns_repeat_aborted', 0)}")
    print(f"  Turns failed (timeout/error):   {l.get('turns_request_failed', 0)}")

    print("\nPer conversation:")
    for row in report["per_file"]:
        med = row["median_latency_per_turn_s"] or 0.0
        print(
            f"  {Path(row['source_file']).name}: "
            f"GT={row['ground_truth_gaps']}, "
            f"Pred={row['qwen_predicted_gaps']}, "
            f"TP={row['true_positives']}, "
            f"FN={row['false_negatives']}, "
            f"FP={row['false_positive_gaps']}, "
            f"TPR={100*row['true_positive_rate_at_threshold']:.2f}%, "
            f"median latency={med:.3f}s"
        )


# ============================================================
# MAIN
# ============================================================

def atomic_write_json(path, obj):
    """Write to a temp file, then rename, so a job killed mid-write
    never leaves a half-written JSON behind."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_report(results, failures):
    results = sorted(results, key=lambda r: r["source_file"])
    report = aggregate(results)
    report["failures"] = failures
    atomic_write_json(OUT_FILE, report)
    return report


def pred_path_for(path):
    relative = path.relative_to(DATASETS_DIR)
    pred_path = PRED_DIR / relative
    return pred_path.with_name(pred_path.stem + ".qwen_info_gap_predictions.json")


def load_finished(path):
    """Return the saved result for this file, or None if it must be (re)run."""
    pred_path = pred_path_for(path)
    if not (RESUME and pred_path.exists()):
        return None
    try:
        with pred_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        if "metrics" in result and "turns" in result:
            return result
    except (json.JSONDecodeError, OSError):
        pass
    print(f"  {path.name}: incomplete prediction file; re-running")
    return None


def run_one(path):
    """Process one conversation file in its own thread with its own session."""
    session = requests.Session()
    try:
        result = process_file(path, session)
    finally:
        session.close()

    pred_path = pred_path_for(path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(pred_path, result)
    return result


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATASETS_DIR.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in:\n{DATASETS_DIR}")

    # --- Few-shot: pick demo conversations, build examples, exclude them ---
    demo_files = pick_fewshot_files(files)
    FEWSHOT_MESSAGES[:], shots_info = build_fewshot_messages(demo_files)
    files = [p for p in files if p not in demo_files]
    FEWSHOT_INFO.update({
        "demo_files_excluded_from_evaluation": [p.name for p in demo_files],
        "examples": shots_info,
        "n_examples": len(shots_info),
        "history_turns_per_example": FEWSHOT_HISTORY_TURNS,
        "seed": FEWSHOT_SEED,
    })
    print(f"Few-shot: {len(shots_info)} examples from {[p.name for p in demo_files]} "
          f"(these files are not scored); evaluating {len(files)} files")


    print(f"Model: {QWEN_MODEL} | thinking: {'ON' if THINK else 'OFF'} | "
          f"Ollama: {OLLAMA_HOST}")
    print(f"Output folder: {PRED_DIR}")

    session = requests.Session()
    check_ollama(session)

    print("Warming Qwen; warm-up is excluded from evaluation latency...")
    warm_up(session)

    results = []
    failures = []

    # --- Resume: load files that already finished in an earlier run ---
    to_run = []
    for path in files:
        done = load_finished(path)
        if done is not None:
            results.append(done)
        else:
            to_run.append(path)

    print(
        f"\n{len(files)} files total: {len(results)} already done (skipped), "
        f"{len(to_run)} to run with {NUM_WORKERS} parallel workers."
    )

    # --- Run the remaining files in parallel ---
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(run_one, path): path for path in to_run}

        for done_no, fut in enumerate(as_completed(futures), start=1):
            path = futures[fut]
            try:
                results.append(fut.result())
                print(f"\n[{done_no}/{len(to_run)}] FINISHED {path.name}")
            except Exception as exc:
                failures.append({"file": str(path), "error": repr(exc)})
                print(f"\n[{done_no}/{len(to_run)}] ERROR in {path.name}: {exc}")

            # checkpoint
            if results:
                write_report(results, failures)

    if not results:
        raise RuntimeError("No files completed successfully.")

    report = write_report(results, failures)
    print_summary(report)
    print(f"\nSaved report:\n{OUT_FILE}")


if __name__ == "__main__":
    main()