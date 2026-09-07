import json
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
    r"\baselines\qwen_detection_predictions"
)

OUT_FILE = PRED_DIR / "eval_detection_report.json"

QWEN_MODEL = "qwen3:4b"
EMBED_MODEL = "qwen3-embedding:0.6b"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# User-requested threshold:
RESOLUTION_THRESHOLD = 0.70

# Used only to match Qwen's detected phrase to the GT ambiguous phrase.
# 1.0 = exact normalized phrase match.
PHRASE_MATCH_THRESHOLD = 0.90

TEMPERATURE = 0.0
SEED = 42
NUM_PREDICT = 512
REQUEST_TIMEOUT_S = 600
KEEP_ALIVE = "30m"


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


def normalize_phrase(s):
    s = str(s or "").strip().lower()
    s = s.replace("*", "")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9'\s-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" -'\"")


def phrase_similarity(a, b):
    a = normalize_phrase(a)
    b = normalize_phrase(b)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    return float(SequenceMatcher(None, a, b).ratio())


def transcript_until(transcript, turn_id):
    rows = [
        t for t in transcript
        if int(t.get("turn_id", -1)) <= int(turn_id)
    ]
    rows.sort(key=lambda x: int(x.get("turn_id", 0)))
    return rows


def format_transcript(rows):
    return "\n".join(
        f"[Turn {t.get('turn_id')}] "
        f"{t.get('speaker', 'Unknown')}: "
        f"{t.get('text', '')}"
        for t in rows
    )


def group_gt_by_turn(gt_rows):
    grouped = defaultdict(list)

    for gt_id, row in enumerate(gt_rows):
        item = dict(row)
        item["_gt_id"] = gt_id
        grouped[int(row["trigger_turn_id"])].append(item)

    return grouped


# ============================================================
# OLLAMA
# ============================================================

def check_ollama(session):
    r = session.get(OLLAMA_TAGS_URL, timeout=15)
    r.raise_for_status()

    names = {
        m.get("name")
        for m in r.json().get("models", [])
        if m.get("name")
    }

    missing = [
        m for m in [QWEN_MODEL, EMBED_MODEL]
        if m not in names
    ]

    if missing:
        raise RuntimeError(
            "Missing Ollama model(s): "
            + ", ".join(missing)
            + "\nRun:\n"
            + "\n".join(f"ollama pull {m}" for m in missing)
        )


def output_schema():
    return {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ambiguous_phrase": {"type": "string"},
                        "resolved_entity": {"type": "string"},
                    },
                    "required": [
                        "ambiguous_phrase",
                        "resolved_entity",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["detections"],
        "additionalProperties": False,
    }


def make_prompt(transcript, snapshot, current_turn):
    turn_id = int(current_turn["turn_id"])

    system = """
You are an ambiguity detection and contextual entity-resolution system.

Evaluate ONLY the CURRENT TURN.

Detect expressions whose intended referent or meaning depends on previous
dialogue or the supplied mobile context. These may include:

- pronouns: he, she, they, it, them, we
- demonstratives: this, that, these, those
- deictic locations: here, there, my place
- relative time: today, tomorrow, later, next Monday
- context-dependent entities: the document, the meeting, the client,
  that feature, the channel, the files, etc.

Rules:
1. Do NOT mark every noun phrase as ambiguous.
2. A phrase should be detected only when its intended identity/reference
   genuinely requires contextual interpretation.
3. `ambiguous_phrase` MUST be copied exactly from the CURRENT TURN.
4. Resolve the phrase using transcript history and mobile_context_snapshot.
5. Do not invent unsupported exact filenames, dates, IDs, or names.
6. If the current turn contains no ambiguous phrase, return {"detections":[]}.
7. Return JSON only.
""".strip()

    payload = {
        "current_turn_id": turn_id,
        "current_speaker": current_turn.get("speaker"),
        "current_turn_text": current_turn.get("text", ""),
        "mobile_context_snapshot": snapshot,
        "conversation_so_far": format_transcript(
            transcript_until(transcript, turn_id)
        ),
    }

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


def call_qwen(session, messages):
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "stream": False,
        "format": output_schema(),
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
        },
    }

    t0 = time.perf_counter()

    r = session.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )

    latency = time.perf_counter() - t0

    if not r.ok:
        raise RuntimeError(
            f"Ollama chat failed ({r.status_code}): {r.text}"
        )

    raw = r.json()
    content = raw.get("message", {}).get("content", "")

    parsed = json.loads(content)
    detections = parsed.get("detections", [])

    cleaned = []

    for d in detections:
        phrase = str(d.get("ambiguous_phrase", "")).strip()
        resolution = str(d.get("resolved_entity", "")).strip()

        if phrase:
            cleaned.append({
                "ambiguous_phrase": phrase,
                "resolved_entity": resolution,
            })

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
        "generation_tokens_per_s": safe_div(
            output_tokens,
            generation_s,
        ),
    }

    return cleaned, compute


def warm_up(session):
    messages = [
        {
            "role": "user",
            "content": 'Return JSON: {"detections":[]}',
        }
    ]

    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "stream": False,
        "format": output_schema(),
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 32,
        },
    }

    r = session.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )

    if not r.ok:
        raise RuntimeError(
            f"Qwen warm-up failed ({r.status_code}): {r.text}"
        )


# ============================================================
# RESOLUTION SIMILARITY
# ============================================================

def resolution_similarities(session, pairs):
    """
    pairs:
        [(predicted_resolution, ground_truth_resolution), ...]
    """
    if not pairs:
        return []

    inputs = []

    for pred, gt in pairs:
        inputs.append(str(pred or ""))
        inputs.append(str(gt or ""))

    r = session.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": inputs,
            "keep_alive": KEEP_ALIVE,
        },
        timeout=REQUEST_TIMEOUT_S,
    )

    if not r.ok:
        raise RuntimeError(
            f"Ollama embed failed ({r.status_code}): {r.text}"
        )

    embs = r.json().get("embeddings", [])

    if len(embs) != len(inputs):
        raise ValueError(
            f"Expected {len(inputs)} embeddings, got {len(embs)}"
        )

    sims = []

    for i in range(0, len(embs), 2):
        a = np.asarray(embs[i], dtype=np.float64)
        b = np.asarray(embs[i + 1], dtype=np.float64)

        denom = np.linalg.norm(a) * np.linalg.norm(b)

        sim = (
            float(np.dot(a, b) / denom)
            if denom else 0.0
        )

        sims.append(max(-1.0, min(1.0, sim)))

    return sims


# ============================================================
# PHRASE MATCHING
# ============================================================

def match_predictions(predictions, gt_items):
    """
    One-to-one matching of predicted ambiguous phrases to GT phrases.

    Returns:
      matches = [(pred_index, gt_index, phrase_score), ...]
      unmatched_pred_indices
      unmatched_gt_indices
    """
    candidates = []

    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(gt_items):
            score = phrase_similarity(
                pred["ambiguous_phrase"],
                gt["ambiguous_phrase"],
            )

            if score >= PHRASE_MATCH_THRESHOLD:
                candidates.append(
                    (score, pi, gi)
                )

    candidates.sort(reverse=True)

    used_p = set()
    used_g = set()
    matches = []

    for score, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue

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

    return matches, unmatched_p, unmatched_g


# ============================================================
# TURN EVALUATION
# ============================================================

def evaluate_turn(
    session,
    current_turn,
    predictions,
    gt_items,
    compute,
):
    matches, unmatched_p, unmatched_g = match_predictions(
        predictions,
        gt_items,
    )

    pairs = [
        (
            predictions[pi]["resolved_entity"],
            gt_items[gi]["resolved_entity"],
        )
        for pi, gi, _ in matches
    ]

    sim_scores = resolution_similarities(
        session,
        pairs,
    )

    matched_rows = []
    tp = 0
    bad_resolution = 0

    for (
        (pi, gi, phrase_score),
        resolution_score,
    ) in zip(matches, sim_scores):

        pred = predictions[pi]
        gt = gt_items[gi]

        is_tp = (
            resolution_score
            >= RESOLUTION_THRESHOLD
        )

        if is_tp:
            tp += 1
        else:
            bad_resolution += 1

        matched_rows.append({
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
                resolution_score,

            "positive_similarity_ge_0_70":
                is_tp,

            "resolution_source":
                gt.get("resolution_source"),
        })

    false_positives = [
        predictions[i]
        for i in unmatched_p
    ]

    missed_gt = [
        {
            "ambiguous_phrase":
                gt_items[i]["ambiguous_phrase"],
            "ground_truth_resolution":
                gt_items[i]["resolved_entity"],
            "resolution_source":
                gt_items[i].get("resolution_source"),
        }
        for i in unmatched_g
    ]

    # User-requested definitions:
    #
    # TP = detected GT phrase + resolution >= 0.70
    #
    # FN = missed GT phrase OR detected GT phrase
    #      whose resolution similarity < 0.70
    #
    # FP = predicted ambiguous phrase that did not match
    #      any GT ambiguous phrase.
    fn = len(unmatched_g) + bad_resolution
    fp = len(unmatched_p)

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
                fn,

            "false_positives":
                fp,

            "missed_gt_ambiguities":
                len(unmatched_g),

            "detected_but_resolution_below_0_70":
                bad_resolution,
        },

        "matched": matched_rows,
        "false_positive_predictions":
            false_positives,
        "missed_ground_truth":
            missed_gt,
        "all_qwen_predictions":
            predictions,
        "compute":
            compute,
    }


# ============================================================
# METRICS
# ============================================================

def metrics_from_turns(turns):
    gt = sum(
        t["counts"]["ground_truth_ambiguities"]
        for t in turns
    )

    pred = sum(
        t["counts"]["predicted_ambiguities"]
        for t in turns
    )

    detected = sum(
        t["counts"]["detected_gt_ambiguities"]
        for t in turns
    )

    tp = sum(
        t["counts"]["true_positives"]
        for t in turns
    )

    fn = sum(
        t["counts"]["false_negatives"]
        for t in turns
    )

    fp = sum(
        t["counts"]["false_positives"]
        for t in turns
    )

    missed = sum(
        t["counts"]["missed_gt_ambiguities"]
        for t in turns
    )

    bad_res = sum(
        t["counts"]["detected_but_resolution_below_0_70"]
        for t in turns
    )

    # Detection only:
    detection_recall = safe_div(
        detected,
        gt,
    )

    detection_precision = safe_div(
        detected,
        detected + fp,
    )

    detection_f1 = safe_div(
        2 * detection_precision * detection_recall,
        detection_precision + detection_recall,
    )

    # End-to-end:
    #
    # If GT has 10 ambiguities, Qwen detects 8,
    # and all 8 resolution scores >= .70:
    # TPR = 8/10 = 80%.
    #
    # If only 6 of those 8 exceed .70:
    # TPR = 6/10 = 60%.
    tpr = safe_div(
        tp,
        gt,
    )

    # Strict precision:
    # Of ALL ambiguous phrases Qwen predicted,
    # how many were correct detections AND had
    # resolution similarity >= .70?
    strict_precision = safe_div(
        tp,
        pred,
    )

    end_to_end_f1 = safe_div(
        2 * strict_precision * tpr,
        strict_precision + tpr,
    )

    # User-requested FP statistic:
    false_positive_discovery_rate = safe_div(
        fp,
        pred,
    )

    return {
        "ground_truth_ambiguities":
            gt,

        "qwen_predicted_ambiguities":
            pred,

        "detected_ground_truth_ambiguities":
            detected,

        "missed_ground_truth_ambiguities":
            missed,

        "detected_but_bad_resolution":
            bad_res,

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
            detection_f1,

        "true_positive_rate_at_0_70":
            tpr,

        "strict_end_to_end_precision_at_0_70":
            strict_precision,

        "end_to_end_f1_at_0_70":
            end_to_end_f1,

        "false_positive_discovery_rate":
            false_positive_discovery_rate,

        "false_positives_per_turn":
            safe_div(fp, len(turns)),

        "classical_false_positive_rate":
            None,

        "fpr_note":
            (
                "Classical FPR = FP/(FP+TN) is not defined "
                "unless a finite set of negative candidate phrases "
                "is specified."
            ),
    }


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
            ),
    }


# ============================================================
# ONE FILE
# ============================================================

def process_file(path, session):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    transcript = data.get(
        "conversation_transcript",
        [],
    )

    snapshot = data.get(
        "mobile_context_snapshot",
        {},
    )

    gt_rows = data.get(
        "ground_truth_resolutions",
        [],
    )

    gt_map = group_gt_by_turn(
        gt_rows
    )

    evaluated_turns = []

    # IMPORTANT:
    # Run Qwen on EVERY turn.
    # Otherwise false positives on turns with
    # no GT ambiguity cannot be measured.
    for i, turn in enumerate(
        transcript,
        start=1,
    ):
        turn_id = int(turn["turn_id"])

        messages = make_prompt(
            transcript,
            snapshot,
            turn,
        )

        predictions, compute = call_qwen(
            session,
            messages,
        )

        evaluated = evaluate_turn(
            session=session,
            current_turn=turn,
            predictions=predictions,
            gt_items=gt_map.get(
                turn_id,
                [],
            ),
            compute=compute,
        )

        evaluated_turns.append(
            evaluated
        )

        c = evaluated["counts"]

        print(
            f"  turn {turn_id:>3} "
            f"({i}/{len(transcript)}) "
            f"GT={c['ground_truth_ambiguities']} "
            f"PRED={c['predicted_ambiguities']} "
            f"TP={c['true_positives']} "
            f"FN={c['false_negatives']} "
            f"FP={c['false_positives']} "
            f"lat={compute['wall_latency_s']:.2f}s"
        )

    return {
        "dataset_id":
            data.get("dataset_id"),

        "source_file":
            str(path),

        "metrics":
            metrics_from_turns(
                evaluated_turns
            ),

        "latency_compute":
            latency_from_turns(
                evaluated_turns
            ),

        "turns":
            evaluated_turns,
    }


# ============================================================
# GLOBAL REPORT
# ============================================================

def aggregate(file_results):
    all_turns = [
        turn
        for result in file_results
        for turn in result["turns"]
    ]

    per_file = []

    for result in file_results:
        per_file.append({
            "dataset_id":
                result["dataset_id"],

            "source_file":
                result["source_file"],

            **result["metrics"],
            **result["latency_compute"],
        })

    overall_latency = latency_from_turns(
        all_turns
    )

    file_medians = [
        x["median_latency_per_turn_s"]
        for x in per_file
        if x["median_latency_per_turn_s"]
        is not None
    ]

    overall_latency[
        "median_of_conversation_median_latencies_s"
    ] = median_or_none(
        file_medians
    )

    return {
        "configuration": {
            "qwen_model":
                QWEN_MODEL,

            "embedding_model":
                EMBED_MODEL,

            "resolution_similarity_threshold":
                RESOLUTION_THRESHOLD,

            "phrase_match_threshold":
                PHRASE_MATCH_THRESHOLD,

            "ground_truth_shown_to_qwen":
                False,
        },

        "overall":
            metrics_from_turns(
                all_turns
            ),

        "latency_compute":
            overall_latency,

        "per_file":
            per_file,
    }


def print_summary(report):
    m = report["overall"]
    l = report["latency_compute"]

    print("\n" + "=" * 80)
    print("AMBIGUITY DETECTION + RESOLUTION")
    print("=" * 80)

    print(
        f"GT ambiguous phrases:             "
        f"{m['ground_truth_ambiguities']}"
    )

    print(
        f"Qwen predicted ambiguities:       "
        f"{m['qwen_predicted_ambiguities']}"
    )

    print(
        f"Detected GT ambiguities:          "
        f"{m['detected_ground_truth_ambiguities']}"
    )

    print(
        f"Missed GT ambiguities:            "
        f"{m['missed_ground_truth_ambiguities']}"
    )

    print(
        f"Detected but similarity < 0.70:   "
        f"{m['detected_but_bad_resolution']}"
    )

    print(
        f"TRUE POSITIVES:                   "
        f"{m['true_positives']}"
    )

    print(
        f"FALSE NEGATIVES:                  "
        f"{m['false_negatives']}"
    )

    print(
        f"FALSE POSITIVES:                  "
        f"{m['false_positive_ambiguities']}"
    )

    print("\nDetection only:")
    print(
        f"  Recall:                         "
        f"{100*m['detection_recall']:.2f}%"
    )
    print(
        f"  Precision:                      "
        f"{100*m['detection_precision']:.2f}%"
    )
    print(
        f"  F1:                             "
        f"{100*m['detection_f1']:.2f}%"
    )

    print("\nDetection + correct resolution:")
    print(
        f"  TPR / Recall @ 0.70:            "
        f"{100*m['true_positive_rate_at_0_70']:.2f}%"
    )
    print(
        f"  Precision @ 0.70:               "
        f"{100*m['strict_end_to_end_precision_at_0_70']:.2f}%"
    )
    print(
        f"  F1 @ 0.70:                      "
        f"{100*m['end_to_end_f1_at_0_70']:.2f}%"
    )

    print("\nFalse positives:")
    print(
        f"  FP discovery rate:              "
        f"{100*m['false_positive_discovery_rate']:.2f}%"
    )
    print(
        f"  FP per turn:                    "
        f"{m['false_positives_per_turn']:.4f}"
    )

    print("\nLatency:")
    print(
        f"  Median turn latency:            "
        f"{l['median_latency_per_turn_s']:.3f}s"
    )

    print(
        f"  Median of file medians:         "
        f"{l['median_of_conversation_median_latencies_s']:.3f}s"
    )

    print("\nPer conversation:")
    for row in report["per_file"]:
        print(
            f"  {Path(row['source_file']).name}: "
            f"GT={row['ground_truth_ambiguities']}, "
            f"Pred={row['qwen_predicted_ambiguities']}, "
            f"TP={row['true_positives']}, "
            f"FN={row['false_negatives']}, "
            f"FP={row['false_positive_ambiguities']}, "
            f"TPR={100*row['true_positive_rate_at_0_70']:.2f}%, "
            f"median latency={row['median_latency_per_turn_s']:.3f}s"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    PRED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        DATASETS_DIR.rglob("*.json")
    )

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in:\n{DATASETS_DIR}"
        )

    session = requests.Session()

    check_ollama(
        session
    )

    print(
        "Warming Qwen; warm-up is excluded "
        "from evaluation latency..."
    )

    warm_up(
        session
    )

    results = []
    failures = []

    for file_no, path in enumerate(
        files,
        start=1,
    ):
        print(
            f"\n[{file_no}/{len(files)}] "
            f"{path.name}"
        )

        try:
            result = process_file(
                path,
                session,
            )

            results.append(
                result
            )

            relative = path.relative_to(
                DATASETS_DIR
            )

            pred_path = (
                PRED_DIR
                / relative
            )

            pred_path = pred_path.with_name(
                pred_path.stem
                + ".qwen_detection_predictions.json"
            )

            pred_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with pred_path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except Exception as exc:
            failures.append({
                "file": str(path),
                "error": repr(exc),
            })

            print(
                f"ERROR: {exc}"
            )

        # checkpoint
        if results:
            report = aggregate(
                results
            )
            report["failures"] = failures

            with OUT_FILE.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    report,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    if not results:
        raise RuntimeError(
            "No files completed successfully."
        )

    report = aggregate(
        results
    )

    report["failures"] = failures

    with OUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_summary(
        report
    )

    print(
        f"\nSaved report:\n{OUT_FILE}"
    )


if __name__ == "__main__":
    main()
