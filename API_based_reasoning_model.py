
import json
import os
import re
import statistics
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import requests


# ============================================================
# CONFIG
# ============================================================

DATASETS_DIR = Path(
    r"C:\Users\amart\Desktop\PhD projects\lexicon-framework"
    r"\data_generation\event_keywords\generated_datasets"
)

PRED_DIR = Path(
    r"C:\Users\amart\Desktop\PhD projects\lexicon-framework"
    r"\baselines\hosted_api_detection_predictions"
)

OUT_FILE = PRED_DIR / "eval_detection_report.json"


# ================= MODEL API =================

MODEL_API_BASE_URL = os.getenv(
    "MODEL_API_BASE_URL",
    "https://api.openai.com/v1"
).rstrip("/")

MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

CHAT_API_URL = os.getenv(
    "CHAT_API_URL",
    f"{MODEL_API_BASE_URL}/chat/completions"
)


# ================= EMBEDDING API =================

EMBED_API_BASE_URL = os.getenv(
    "EMBED_API_BASE_URL",
    MODEL_API_BASE_URL
).rstrip("/")

EMBED_API_KEY = os.getenv(
    "EMBED_API_KEY",
    MODEL_API_KEY
)

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "text-embedding-3-small"
)

EMBED_API_URL = os.getenv(
    "EMBED_API_URL",
    f"{EMBED_API_BASE_URL}/embeddings"
)


# ================= EVALUATION =================

RESOLUTION_THRESHOLD = 0.70
PHRASE_MATCH_THRESHOLD = 0.90

TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_TOKENS_PARAMETER = "max_tokens"

# Options: "json_object", "json_schema", "none"
RESPONSE_FORMAT_MODE = "json_object"

REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 3


# ============================================================
# HELPERS
# ============================================================

def safe_div(a, b):
    return float(a / b) if b else 0.0


def median_or_none(values):
    return (
        float(statistics.median(values))
        if values else None
    )


def mean_or_none(values):
    return (
        float(statistics.mean(values))
        if values else None
    )


def normalize_phrase(text):
    text = str(text or "").strip().lower()
    text = text.replace("*", "")
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9'\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" -'\"")


def phrase_similarity(a, b):
    a = normalize_phrase(a)
    b = normalize_phrase(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return float(
        SequenceMatcher(None, a, b).ratio()
    )


def group_gt_by_turn(gt_rows):
    grouped = defaultdict(list)

    for item in gt_rows:
        grouped[
            int(item["trigger_turn_id"])
        ].append(item)

    return grouped


# ============================================================
# OUTPUT SCHEMA
# ============================================================

def output_schema():
    return {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ambiguous_phrase": {
                            "type": "string"
                        },
                        "resolved_entity": {
                            "type": "string"
                        },
                    },
                    "required": [
                        "ambiguous_phrase",
                        "resolved_entity"
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["detections"],
        "additionalProperties": False,
    }


# ============================================================
# PROMPT
# ============================================================

def make_prompt(transcript, snapshot, current_turn):

    turn_id = int(current_turn["turn_id"])

    prior_and_current = sorted(
        (
            t for t in transcript
            if int(t.get("turn_id", -1)) <= turn_id
        ),
        key=lambda t: int(t["turn_id"]),
    )

    history = "\n".join(
        f"[Turn {t.get('turn_id')}] "
        f"{t.get('speaker', 'Unknown')}: "
        f"{t.get('text', '')}"
        for t in prior_and_current
    )

    system = """
You are an ambiguity detection and contextual entity-resolution system.

Evaluate ONLY the CURRENT TURN.

Detect expressions whose intended referent or meaning depends on previous
conversation or the supplied mobile context.

Examples:
- pronouns: he, she, they, it, them, we
- demonstratives: this, that, these, those
- deictic locations: here, there, my place
- relative time: today, tomorrow, later, next Monday
- context-dependent entities: the document, the meeting, the client, etc.

Rules:
1. Do NOT mark every noun phrase as ambiguous.
2. Detect a phrase only when its intended reference genuinely needs context.
3. Ambiguous_phrase MUST be copied exactly from the CURRENT TURN.
4. Resolve using conversation history and mobile_context_snapshot.
5. Do not invent unsupported exact filenames, dates, IDs or names.
6. If no ambiguity is present, return {"detections":[]}.
7. Return JSON only in the format:
   {"detections":[{"ambiguous_phrase":"...","resolved_entity":"..."}]}
""".strip()

    payload = {
        "current_turn_id": turn_id,
        "current_speaker": current_turn.get("speaker"),
        "current_turn_text": current_turn.get("text", ""),
        "mobile_context_snapshot": snapshot,
        "conversation_so_far": history,
    }

    return [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                indent=2
            )
        }
    ]


# ============================================================
# API REQUEST
# ============================================================

def post_api(session, url, api_key, payload, label):

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES + 1):

        try:
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_S
            )

        except (
            requests.Timeout,
            requests.ConnectionError
        ):
            if attempt == MAX_RETRIES:
                raise

            time.sleep(min(2 ** attempt, 15))
            continue

        if (
            response.status_code in (
                429, 500, 502, 503, 504
            )
            and attempt < MAX_RETRIES
        ):
            time.sleep(min(2 ** attempt, 15))
            continue

        if not response.ok:
            raise RuntimeError(
                f"{label} failed "
                f"({response.status_code}): "
                f"{response.text[:1500]}"
            )

        return response.json()

    raise RuntimeError(
        f"{label} failed after all retries"
    )


# ============================================================
# PARSE MODEL DETECTIONS
# ============================================================

def parse_detections(content):

    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )

    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            "The chat API returned empty/non-text content"
        )

    content = content.strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.I
        )
        content = re.sub(
            r"\s*```$",
            "",
            content
        )

    obj = json.loads(content)

    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("detections"), list)
    ):
        raise ValueError(
            "Expected a JSON object containing a detections array"
        )

    cleaned = []

    for item in obj["detections"]:

        if not isinstance(item, dict):
            continue

        phrase = str(
            item.get("ambiguous_phrase", "")
        ).strip()

        if phrase:
            cleaned.append({
                "ambiguous_phrase": phrase,
                "resolved_entity": str(
                    item.get("resolved_entity", "")
                ).strip()
            })

    return cleaned


# ============================================================
# HOSTED MODEL INFERENCE
# ============================================================

def call_model_api(session, messages):

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        MAX_TOKENS_PARAMETER: MAX_TOKENS
    }

    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE

    if RESPONSE_FORMAT_MODE == "json_object":

        payload["response_format"] = {
            "type": "json_object"
        }

    elif RESPONSE_FORMAT_MODE == "json_schema":

        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ambiguity_detections",
                "strict": True,
                "schema": output_schema()
            }
        }

    elif RESPONSE_FORMAT_MODE != "none":
        raise ValueError(
            "RESPONSE_FORMAT_MODE must be json_object, "
            "json_schema or none"
        )

    t0 = time.perf_counter()

    raw = post_api(
        session,
        CHAT_API_URL,
        MODEL_API_KEY,
        payload,
        "Chat API"
    )

    wall_latency = time.perf_counter() - t0

    choices = raw.get("choices") or []

    if not choices:
        raise ValueError(
            "Chat API returned no choices"
        )

    if choices[0].get("finish_reason") == "length":
        raise ValueError(
            "Chat completion was truncated; increase MAX_TOKENS"
        )

    detections = parse_detections(
        (choices[0].get("message") or {}).get("content")
    )

    usage = raw.get("usage") or {}

    prompt_tokens = int(
        usage.get("prompt_tokens") or 0
    )

    output_tokens = int(
        usage.get("completion_tokens") or 0
    )

    compute = {
        "wall_latency_s": wall_latency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(
            usage.get("total_tokens")
            or (prompt_tokens + output_tokens)
        ),
        "generation_tokens_per_wall_s": safe_div(
            output_tokens,
            wall_latency
        ),
        "provider_model": raw.get(
            "model",
            MODEL_NAME
        )
    }

    return detections, compute


# ============================================================
# RESOLUTION SIMILARITY
# ============================================================

def resolution_similarities(session, pairs):

    if not pairs:
        return []

    inputs = []

    for predicted, expected in pairs:

        inputs.extend([
            str(predicted or "").strip() or " ",
            str(expected or "").strip() or " "
        ])

    raw = post_api(
        session,
        EMBED_API_URL,
        EMBED_API_KEY,
        {
            "model": EMBED_MODEL_NAME,
            "input": inputs
        },
        "Embeddings API"
    )

    rows = sorted(
        raw.get("data") or [],
        key=lambda x: x.get("index", 0)
    )

    if len(rows) != len(inputs):
        raise ValueError(
            f"Expected {len(inputs)} embeddings, "
            f"got {len(rows)}"
        )

    vectors = [
        np.asarray(
            row["embedding"],
            dtype=np.float64
        )
        for row in rows
    ]

    result = []

    for i in range(0, len(vectors), 2):

        a = vectors[i]
        b = vectors[i + 1]

        norm = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        value = (
            float(np.dot(a, b) / norm)
            if norm else 0.0
        )

        result.append(
            max(-1.0, min(1.0, value))
        )

    return result


# ============================================================
# PHRASE MATCHING
# ============================================================

def match_predictions(predictions, gt_items):

    candidates = [
        (
            phrase_similarity(
                p["ambiguous_phrase"],
                g["ambiguous_phrase"]
            ),
            pi,
            gi
        )
        for pi, p in enumerate(predictions)
        for gi, g in enumerate(gt_items)
    ]

    candidates = sorted(
        (
            x for x in candidates
            if x[0] >= PHRASE_MATCH_THRESHOLD
        ),
        reverse=True
    )

    used_p = set()
    used_g = set()
    matches = []

    for score, pi, gi in candidates:

        if pi not in used_p and gi not in used_g:

            used_p.add(pi)
            used_g.add(gi)

            matches.append(
                (pi, gi, score)
            )

    unmatched_p = [
        i for i in range(len(predictions))
        if i not in used_p
    ]

    unmatched_g = [
        i for i in range(len(gt_items))
        if i not in used_g
    ]

    return (
        matches,
        unmatched_p,
        unmatched_g
    )


# ============================================================
# TURN EVALUATION
# ============================================================

def evaluate_turn(
    session,
    current_turn,
    predictions,
    gt_items,
    compute
):

    matches, unmatched_p, unmatched_g = match_predictions(
        predictions,
        gt_items
    )

    pairs = [
        (
            predictions[pi]["resolved_entity"],
            gt_items[gi]["resolved_entity"]
        )
        for pi, gi, _ in matches
    ]

    similarities = resolution_similarities(
        session,
        pairs
    )

    matched = []
    tp = 0
    bad_res = 0

    for (
        (pi, gi, phrase_score),
        sim
    ) in zip(matches, similarities):

        pred = predictions[pi]
        gt = gt_items[gi]

        correct = (
            sim >= RESOLUTION_THRESHOLD
        )

        tp += int(correct)
        bad_res += int(not correct)

        matched.append({
            "predicted_ambiguous_phrase":
                pred["ambiguous_phrase"],
            "ground_truth_ambiguous_phrase":
                gt["ambiguous_phrase"],
            "phrase_match_score":
                phrase_score,
            "predicted_resolution":
                pred["resolved_entity"],
            "ground_truth_resolution":
                gt["resolved_entity"],
            "resolution_similarity":
                sim,
            "positive_similarity_ge_0_70":
                correct,
            "resolution_source":
                gt.get("resolution_source")
        })

    missed = [
        {
            "ambiguous_phrase":
                gt_items[i]["ambiguous_phrase"],
            "ground_truth_resolution":
                gt_items[i]["resolved_entity"],
            "resolution_source":
                gt_items[i].get("resolution_source")
        }
        for i in unmatched_g
    ]

    return {
        "turn_id": int(current_turn["turn_id"]),
        "speaker": current_turn.get("speaker"),
        "text": current_turn.get("text", ""),

        "counts": {
            "ground_truth_ambiguities":
                len(gt_items),

            "predicted_ambiguities":
                len(predictions),

            "detected_gt_ambiguities":
                len(matches),

            "true_positives":
                tp,

            "false_negatives":
                len(unmatched_g) + bad_res,

            "false_positives":
                len(unmatched_p),

            "missed_gt_ambiguities":
                len(unmatched_g),

            "detected_but_resolution_below_0_70":
                bad_res
        },

        "matched": matched,

        "false_positive_predictions": [
            predictions[i]
            for i in unmatched_p
        ],

        "missed_ground_truth": missed,

        "all_model_predictions": predictions,

        "compute": compute
    }


# ============================================================
# METRICS
# ============================================================

def metrics_from_turns(turns):

    def total(key):
        return sum(
            t["counts"][key]
            for t in turns
        )

    gt = total("ground_truth_ambiguities")
    pred = total("predicted_ambiguities")
    detected = total("detected_gt_ambiguities")

    tp = total("true_positives")
    fn = total("false_negatives")
    fp = total("false_positives")

    detection_recall = safe_div(
        detected,
        gt
    )

    detection_precision = safe_div(
        detected,
        detected + fp
    )

    end_to_end_recall = safe_div(
        tp,
        gt
    )

    end_to_end_precision = safe_div(
        tp,
        pred
    )

    return {
        "ground_truth_ambiguities":
            gt,

        "model_predicted_ambiguities":
            pred,

        "detected_ground_truth_ambiguities":
            detected,

        "missed_ground_truth_ambiguities":
            total("missed_gt_ambiguities"),

        "detected_but_bad_resolution":
            total("detected_but_resolution_below_0_70"),

        "true_positives":
            tp,

        "false_negatives":
            fn,

        "false_positive_ambiguities":
            fp,

        "detection_recall":
            detection_recall,

        "detection_precision":
            detection_precision,

        "detection_f1":
            safe_div(
                2 * detection_precision * detection_recall,
                detection_precision + detection_recall
            ),

        "true_positive_rate_at_0_70":
            end_to_end_recall,

        "strict_end_to_end_precision_at_0_70":
            end_to_end_precision,

        "end_to_end_f1_at_0_70":
            safe_div(
                2 * end_to_end_precision * end_to_end_recall,
                end_to_end_precision + end_to_end_recall
            ),

        "false_positive_discovery_rate":
            safe_div(fp, pred),

        "false_positives_per_turn":
            safe_div(fp, len(turns)),

        "classical_false_positive_rate":
            None,

        "fpr_note":
            "Classical FPR = FP/(FP+TN) requires defined negative candidates."
    }


# ============================================================
# LATENCY
# ============================================================

def latency_from_turns(turns):

    latencies = [
        t["compute"]["wall_latency_s"]
        for t in turns
    ]

    return {
        "number_of_turn_calls":
            len(turns),

        "median_latency_per_turn_s":
            median_or_none(latencies),

        "mean_latency_per_turn_s":
            mean_or_none(latencies),

        "total_prompt_tokens":
            sum(
                t["compute"]["prompt_tokens"]
                for t in turns
            ),

        "total_output_tokens":
            sum(
                t["compute"]["output_tokens"]
                for t in turns
            )
    }


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(path, session):

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    transcript = data.get(
        "conversation_transcript",
        []
    )

    context = data.get(
        "mobile_context_snapshot",
        {}
    )

    ground_truth = group_gt_by_turn(
        data.get(
            "ground_truth_resolutions",
            []
        )
    )

    evaluated_turns = []

    # Evaluate every turn, including turns with no GT ambiguity.

    for i, turn in enumerate(transcript, start=1):

        turn_id = int(turn["turn_id"])

        messages = make_prompt(
            transcript,
            context,
            turn
        )

        predictions, compute = call_model_api(
            session,
            messages
        )

        result = evaluate_turn(
            session,
            turn,
            predictions,
            ground_truth.get(turn_id, []),
            compute
        )

        evaluated_turns.append(result)

        counts = result["counts"]

        print(
            f"  turn {turn_id:>3} "
            f"({i}/{len(transcript)}) "
            f"GT={counts['ground_truth_ambiguities']} "
            f"PRED={counts['predicted_ambiguities']} "
            f"TP={counts['true_positives']} "
            f"FN={counts['false_negatives']} "
            f"FP={counts['false_positives']} "
            f"lat={compute['wall_latency_s']:.2f}s"
        )

    return {
        "dataset_id":
            data.get("dataset_id"),

        "source_file":
            str(path),

        "metrics":
            metrics_from_turns(evaluated_turns),

        "latency_compute":
            latency_from_turns(evaluated_turns),

        "turns":
            evaluated_turns
    }


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(file_results):

    all_turns = [
        turn
        for result in file_results
        for turn in result["turns"]
    ]

    per_file = [
        {
            "dataset_id": item["dataset_id"],
            "source_file": item["source_file"],
            **item["metrics"],
            **item["latency_compute"]
        }
        for item in file_results
    ]

    latency = latency_from_turns(all_turns)

    latency[
        "median_of_conversation_median_latencies_s"
    ] = median_or_none([
        row["median_latency_per_turn_s"]
        for row in per_file
        if row["median_latency_per_turn_s"] is not None
    ])

    return {
        "configuration": {
            "chat_model":
                MODEL_NAME,

            "embedding_model":
                EMBED_MODEL_NAME,

            "chat_api_url":
                CHAT_API_URL,

            "embedding_api_url":
                EMBED_API_URL,

            "resolution_similarity_threshold":
                RESOLUTION_THRESHOLD,

            "phrase_match_threshold":
                PHRASE_MATCH_THRESHOLD,

            "ground_truth_shown_to_model":
                False
        },

        "overall":
            metrics_from_turns(all_turns),

        "latency_compute":
            latency,

        "per_file":
            per_file
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(report):

    m = report["overall"]
    lat = report["latency_compute"]

    print("\n" + "=" * 72)
    print("AMBIGUITY DETECTION + RESOLUTION")
    print("=" * 72)

    fields = [
        ("GT ambiguous phrases",
         "ground_truth_ambiguities"),

        ("Model predicted ambiguities",
         "model_predicted_ambiguities"),

        ("Detected GT ambiguities",
         "detected_ground_truth_ambiguities"),

        ("Missed GT ambiguities",
         "missed_ground_truth_ambiguities"),

        ("Detected but similarity below 0.70",
         "detected_but_bad_resolution"),

        ("TRUE POSITIVES",
         "true_positives"),

        ("FALSE NEGATIVES",
         "false_negatives"),

        ("FALSE POSITIVES",
         "false_positive_ambiguities")
    ]

    for label, key in fields:
        print(f"{label:38}: {m[key]}")

    print("\nDetection only:")

    for label, key in [
        ("Recall", "detection_recall"),
        ("Precision", "detection_precision"),
        ("F1", "detection_f1")
    ]:
        print(
            f"  {label:18}: {100 * m[key]:.2f}%"
        )

    print("\nDetection + correct resolution:")

    for label, key in [
        ("TPR / Recall @ 0.70",
         "true_positive_rate_at_0_70"),

        ("Precision @ 0.70",
         "strict_end_to_end_precision_at_0_70"),

        ("F1 @ 0.70",
         "end_to_end_f1_at_0_70")
    ]:
        print(
            f"  {label:24}: {100 * m[key]:.2f}%"
        )

    print(
        f"\nFP discovery rate: "
        f"{100 * m['false_positive_discovery_rate']:.2f}%"
    )

    print(
        f"FP per turn: "
        f"{m['false_positives_per_turn']:.4f}"
    )

    if lat["median_latency_per_turn_s"] is not None:
        print(
            f"Median turn latency: "
            f"{lat['median_latency_per_turn_s']:.3f}s"
        )

    if lat[
        "median_of_conversation_median_latencies_s"
    ] is not None:

        print(
            "Median of conversation median latencies: "
            f"{lat['median_of_conversation_median_latencies_s']:.3f}s"
        )

    print("\nPer conversation:")

    for row in report["per_file"]:

        print(
            f"  {Path(row['source_file']).name}: "
            f"GT={row['ground_truth_ambiguities']}, "
            f"Pred={row['model_predicted_ambiguities']}, "
            f"TP={row['true_positives']}, "
            f"FN={row['false_negatives']}, "
            f"FP={row['false_positive_ambiguities']}, "
            f"TPR={100 * row['true_positive_rate_at_0_70']:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_API_KEY:
        raise RuntimeError(
            "Set the MODEL_API_KEY environment variable first"
        )

    if not EMBED_API_KEY:
        raise RuntimeError(
            "Set EMBED_API_KEY, or use MODEL_API_KEY for both APIs"
        )

    files = sorted(
        DATASETS_DIR.rglob("*.json")
    )

    if not files:
        raise FileNotFoundError(
            f"No JSON files found under {DATASETS_DIR}"
        )

    PRED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Chat model: {MODEL_NAME} ({CHAT_API_URL})"
    )

    print(
        f"Embedding model: {EMBED_MODEL_NAME} ({EMBED_API_URL})"
    )

    session = requests.Session()

    results = []
    failures = []

    for index, path in enumerate(files, start=1):

        print(
            f"\n[{index}/{len(files)}] {path.name}"
        )

        try:
            result = process_file(
                path,
                session
            )

            results.append(result)

            relative = path.relative_to(
                DATASETS_DIR
            )

            pred_path = (
                PRED_DIR / relative
            ).with_name(
                relative.stem
                + ".api_detection_predictions.json"
            )

            pred_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with pred_path.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        except Exception as exc:

            failures.append({
                "file": str(path),
                "error": repr(exc)
            })

            print(f"ERROR: {exc}")

        # Save a checkpoint after every dataset.

        report = aggregate(results)
        report["failures"] = failures

        with OUT_FILE.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False
            )

    if not results:
        raise RuntimeError(
            f"No datasets completed successfully; see {OUT_FILE}"
        )

    print_summary(report)

    print(
        f"\nSaved report: {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
