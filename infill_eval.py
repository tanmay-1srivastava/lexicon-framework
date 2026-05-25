

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util


# =========================
# CONFIG
# =========================
DATASETS_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\data_generation\event_keywords\generated_datasets"
PRED_DIR     = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\ilm_infill_predictions"
OUT_FILE     = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\ilm_infill_predictions\eval_report.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PRED_SUFFIX = "_ilm_infill.json"

# Drop rows where the extracted infill is empty AND the ground truth was empty.
SKIP_EMPTY = True

# Tokens we strip from extracted infill candidates before comparing.
STRIP_CHARS = ' \t\n.,;:!?"\'`()[]{}—–-'


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


# =========================
# Re-mask the ground truth so we know blank positions
# =========================
def mask_phrases(text: str, phrases: List[str]) -> Tuple[str, List[str]]:
    """
    Replace each phrase's FIRST remaining occurrence with a sentinel marker
    we can find again later. Returns (masked_text_with_sentinels, ordered_phrases_used).

    Sentinel: '@@BLANK_i@@' (i = 0,1,2,...) so we can locate each blank.
    """
    masked = text
    used: List[str] = []
    for i, phrase in enumerate(phrases):
        if not isinstance(phrase, str) or not phrase:
            continue
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        new_masked, n = pattern.subn(f"@@BLANK_{len(used)}@@", masked, count=1)
        if n > 0:
            masked = new_masked
            used.append(phrase)
    return masked, used


# =========================
# Extract what the model wrote at each blank
# =========================
def split_around_blanks(masked_with_sentinels: str) -> Tuple[List[str], int]:
    """
    Split the sentinel-marked text by sentinels in order.
    Returns (segments, n_blanks). len(segments) == n_blanks + 1.
    """
    sentinel_re = re.compile(r"@@BLANK_\d+@@")
    segments = sentinel_re.split(masked_with_sentinels)
    n_blanks = len(segments) - 1
    return segments, n_blanks


def extract_infill_candidates(
    prediction_text: str,
    segments: List[str],
) -> List[str]:
    """
    Use the constant segments around the blanks as anchors to find what was
    placed at each blank in the prediction.

    Algorithm: walk the prediction left-to-right, locating each anchor in
    sequence. Whatever appears BETWEEN anchors is the infill for that blank.
    """
    pred = prediction_text
    n_blanks = len(segments) - 1
    found: List[str] = []

    # Normalize the prediction by stripping the speaker label prefix if present
    # so anchor matching is more reliable. We strip `User X: "` and trailing `"`.
    pred_clean = pred
    m = re.match(r'^\s*User\s+[A-Z]:\s*"', pred_clean)
    if m:
        pred_clean = pred_clean[m.end():]
        # also drop trailing closing quote if any
        pred_clean = re.sub(r'"\s*$', "", pred_clean)

    # Likewise normalize the segments (they came from the bare turn text already,
    # so this only matters if the predicted text has different quoting).
    # We trim leading/trailing whitespace on each segment to make matching loose.
    seg_clean = [s.strip() for s in segments]

    cursor = 0
    for i in range(n_blanks):
        left = seg_clean[i]
        right = seg_clean[i + 1]

        # Find the left anchor starting at cursor
        if left:
            li = pred_clean.find(left, cursor)
            if li == -1:
                # Anchor missing — model may have rewritten the surrounding text.
                # Fall back to empty infill.
                found.append("")
                continue
            blank_start = li + len(left)
        else:
            blank_start = cursor

        # Find the right anchor starting AFTER the left anchor
        if right:
            ri = pred_clean.find(right, blank_start)
            if ri == -1:
                # No closing anchor — take everything to end of string
                ri = len(pred_clean)
            blank_end = ri
        else:
            blank_end = len(pred_clean)

        candidate = pred_clean[blank_start:blank_end]
        found.append(candidate.strip(STRIP_CHARS).strip())

        cursor = blank_end

    # Pad if we somehow ended with fewer
    while len(found) < n_blanks:
        found.append("")
    return found


# =========================
# Metrics
# =========================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def exact_match(a: str, b: str) -> float:
    return 1.0 if normalize_text(a) == normalize_text(b) and a.strip() != "" else 0.0


def token_f1(a: str, b: str) -> float:
    """
    Token-level F1 between two strings (set-based).
    Returns 0.0 if either is empty.
    """
    aa = set(normalize_text(a).split())
    bb = set(normalize_text(b).split())
    if not aa or not bb:
        return 0.0
    inter = aa & bb
    if not inter:
        return 0.0
    p = len(inter) / len(aa)
    r = len(inter) / len(bb)
    return 2 * p * r / (p + r)


def aggregate(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


# =========================
# Main eval per file
# =========================
def evaluate_one_file(
    dataset_path: Path,
    pred_path: Path,
    model: SentenceTransformer,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (per_row_records, per_file_aggregate).
    """
    dataset = load_json(dataset_path)
    pred_doc = load_json(pred_path)

    # Index dataset turns by turn_id
    turns = dataset.get("conversation_transcript", [])
    turn_by_id: Dict[int, Dict[str, Any]] = {}
    for t in turns:
        try:
            turn_by_id[int(t.get("turn_id"))] = t
        except Exception:
            continue

    # Group resolutions by turn (preserve order)
    resolutions = dataset.get("ground_truth_resolutions", []) or []
    res_by_turn: Dict[int, List[Dict[str, Any]]] = {}
    for r in resolutions:
        try:
            tid = int(r.get("trigger_turn_id"))
        except Exception:
            continue
        res_by_turn.setdefault(tid, []).append(r)

    pred_results = pred_doc.get("results", []) or []

    rows: List[Dict[str, Any]] = []

    # Match each prediction result to a (turn_id, resolutions) pair.
    # We rely on ORDER: results are emitted in sorted(turn_id) order by the
    # prediction script, so we re-derive the same ordering here.
    sorted_tids = sorted(res_by_turn.keys())

    for tid, pred_entry in zip(sorted_tids, pred_results):
        if tid not in turn_by_id:
            continue
        gt_turn = turn_by_id[tid]
        # Mirror the prediction script's transcript normalization (strip asterisks)
        gt_text_raw = str(gt_turn.get("text", "")).replace("*", "")

        rez_list = res_by_turn[tid]
        phrases = [r.get("ambiguous_phrase", "") for r in rez_list]
        entities = [r.get("resolved_entity", "") for r in rez_list]

        # Re-mask the GT to learn blank positions
        sentineled, used_phrases = mask_phrases(gt_text_raw, phrases)
        segments, n_blanks = split_around_blanks(sentineled)

        # Only the first len(used_phrases) entries are usable for alignment
        used_phrases_idx = []
        # Find each used phrase's index in the original phrases list
        # (in case some weren't found in the text)
        cursor = 0
        for p in used_phrases:
            for j in range(cursor, len(phrases)):
                if phrases[j] == p:
                    used_phrases_idx.append(j)
                    cursor = j + 1
                    break

        decoded_preds = pred_entry.get("decoded_predictions", []) or []

        if n_blanks == 0 or not decoded_preds:
            # Nothing to compare
            continue

        for sample_idx, pred_text in enumerate(decoded_preds):
            extracted = extract_infill_candidates(pred_text, segments)

            for blank_i in range(n_blanks):
                gt_phrase = used_phrases[blank_i]
                # Map back to the entity for this phrase
                if blank_i < len(used_phrases_idx):
                    entity = entities[used_phrases_idx[blank_i]]
                else:
                    entity = ""
                extracted_word = extracted[blank_i] if blank_i < len(extracted) else ""

                if SKIP_EMPTY and not extracted_word and not entity and not gt_phrase:
                    continue

                rows.append({
                    "file": pred_doc.get("file", pred_path.name),
                    "turn_id": tid,
                    "sample_idx": sample_idx,
                    "blank_idx": blank_i,
                    "ambiguous_phrase": gt_phrase,
                    "resolved_entity": entity,
                    "extracted_infill": extracted_word,
                    # vs literal ambiguous phrase
                    "em_vs_phrase": exact_match(extracted_word, gt_phrase),
                    "f1_vs_phrase": token_f1(extracted_word, gt_phrase),
                    # vs resolved entity
                    "em_vs_entity": exact_match(extracted_word, entity),
                    "f1_vs_entity": token_f1(extracted_word, entity),
                })

    # --- Per-file embedding similarity (one batched encode call) ---
    if rows:
        # vs ambiguous_phrase
        a1 = [r["extracted_infill"] or " " for r in rows]
        a2 = [r["ambiguous_phrase"] or " " for r in rows]
        emb_a1 = model.encode(a1, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        emb_a2 = model.encode(a2, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        sim_phrase = util.cos_sim(emb_a1, emb_a2).cpu().numpy()
        sim_phrase = np.diag(sim_phrase)

        # vs resolved_entity
        b2 = [r["resolved_entity"] or " " for r in rows]
        emb_b2 = model.encode(b2, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        sim_entity = util.cos_sim(emb_a1, emb_b2).cpu().numpy()
        sim_entity = np.diag(sim_entity)

        for i, r in enumerate(rows):
            r["semantic_sim_vs_phrase"] = float(sim_phrase[i])
            r["semantic_sim_vs_entity"] = float(sim_entity[i])

    # --- Per-file aggregate ---
    if rows:
        arr_em_p = np.array([r["em_vs_phrase"] for r in rows])
        arr_f1_p = np.array([r["f1_vs_phrase"] for r in rows])
        arr_em_e = np.array([r["em_vs_entity"] for r in rows])
        arr_f1_e = np.array([r["f1_vs_entity"] for r in rows])
        arr_sem_p = np.array([r["semantic_sim_vs_phrase"] for r in rows])
        arr_sem_e = np.array([r["semantic_sim_vs_entity"] for r in rows])
    else:
        arr_em_p = arr_f1_p = arr_em_e = arr_f1_e = arr_sem_p = arr_sem_e = np.array([])

    per_file = {
        "file": pred_doc.get("file", pred_path.name),
        "n_rows": len(rows),
        "vs_ambiguous_phrase": {
            "exact_match": aggregate(arr_em_p),
            "token_f1": aggregate(arr_f1_p),
            "semantic_similarity": aggregate(arr_sem_p),
        },
        "vs_resolved_entity": {
            "exact_match": aggregate(arr_em_e),
            "token_f1": aggregate(arr_f1_e),
            "semantic_similarity": aggregate(arr_sem_e),
        },
    }
    return rows, per_file


# =========================
# Main
# =========================
def main() -> Dict[str, Any]:
    datasets_dir = Path(DATASETS_DIR)
    pred_dir = Path(PRED_DIR)

    if not datasets_dir.exists():
        raise RuntimeError(f"DATASETS_DIR not found: {datasets_dir}")
    if not pred_dir.exists():
        raise RuntimeError(f"PRED_DIR not found: {pred_dir}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    pred_files = sorted(pred_dir.glob(f"*{PRED_SUFFIX}"))
    if not pred_files:
        raise RuntimeError(f"No prediction files matched *{PRED_SUFFIX} in {pred_dir}")

    all_rows: List[Dict[str, Any]] = []
    per_file_records: List[Dict[str, Any]] = []

    for pred_path in pred_files:
        # Reverse-engineer the dataset filename from the prediction stem
        stem = pred_path.name[: -len(PRED_SUFFIX)]
        dataset_path = datasets_dir / (stem + ".json")
        if not dataset_path.exists():
            print(f"[skip] Could not find dataset for {pred_path.name}")
            continue

        try:
            rows, per_file = evaluate_one_file(dataset_path, pred_path, model)
            all_rows.extend(rows)
            per_file_records.append(per_file)
            print(f"{pred_path.name} -> {per_file['n_rows']} rows")
        except Exception as e:
            print(f"{pred_path.name} -> FAILED: {e}")

    # --- Global aggregate ---
    if all_rows:
        arr_em_p = np.array([r["em_vs_phrase"] for r in all_rows])
        arr_f1_p = np.array([r["f1_vs_phrase"] for r in all_rows])
        arr_em_e = np.array([r["em_vs_entity"] for r in all_rows])
        arr_f1_e = np.array([r["f1_vs_entity"] for r in all_rows])
        arr_sem_p = np.array([r["semantic_sim_vs_phrase"] for r in all_rows])
        arr_sem_e = np.array([r["semantic_sim_vs_entity"] for r in all_rows])
    else:
        arr_em_p = arr_f1_p = arr_em_e = arr_f1_e = arr_sem_p = arr_sem_e = np.array([])

    report = {
        "datasets_dir": str(datasets_dir),
        "pred_dir": str(pred_dir),
        "embedding_model": EMBED_MODEL_NAME,
        "n_files": len(per_file_records),
        "n_rows_total": len(all_rows),
        "global": {
            "vs_ambiguous_phrase": {
                "exact_match": aggregate(arr_em_p),
                "token_f1": aggregate(arr_f1_p),
                "semantic_similarity": aggregate(arr_sem_p),
            },
            "vs_resolved_entity": {
                "exact_match": aggregate(arr_em_e),
                "token_f1": aggregate(arr_f1_e),
                "semantic_similarity": aggregate(arr_sem_e),
            },
        },
        "per_file": per_file_records,
        "rows": all_rows,
    }

    save_json(Path(OUT_FILE), report)

    # --- Pretty print summary ---
    print("\n================ GLOBAL SUMMARY ================")
    g = report["global"]
    print(f"Total rows: {report['n_rows_total']}")
    print("\nvs Ambiguous Phrase (did the model just copy back the pronoun/phrase?):")
    print(f"  Exact match : mean={g['vs_ambiguous_phrase']['exact_match']['mean']:.4f}  "
          f"median={g['vs_ambiguous_phrase']['exact_match']['median']:.4f}  "
          f"std={g['vs_ambiguous_phrase']['exact_match']['std']:.4f}")
    print(f"  Token F1    : mean={g['vs_ambiguous_phrase']['token_f1']['mean']:.4f}  "
          f"median={g['vs_ambiguous_phrase']['token_f1']['median']:.4f}  "
          f"std={g['vs_ambiguous_phrase']['token_f1']['std']:.4f}")
    print(f"  Semantic    : mean={g['vs_ambiguous_phrase']['semantic_similarity']['mean']:.4f}  "
          f"median={g['vs_ambiguous_phrase']['semantic_similarity']['median']:.4f}  "
          f"std={g['vs_ambiguous_phrase']['semantic_similarity']['std']:.4f}")
    print("\nvs Resolved Entity (did the model produce the actual referent?):")
    print(f"  Exact match : mean={g['vs_resolved_entity']['exact_match']['mean']:.4f}  "
          f"median={g['vs_resolved_entity']['exact_match']['median']:.4f}  "
          f"std={g['vs_resolved_entity']['exact_match']['std']:.4f}")
    print(f"  Token F1    : mean={g['vs_resolved_entity']['token_f1']['mean']:.4f}  "
          f"median={g['vs_resolved_entity']['token_f1']['median']:.4f}  "
          f"std={g['vs_resolved_entity']['token_f1']['std']:.4f}")
    print(f"  Semantic    : mean={g['vs_resolved_entity']['semantic_similarity']['mean']:.4f}  "
          f"median={g['vs_resolved_entity']['semantic_similarity']['median']:.4f}  "
          f"std={g['vs_resolved_entity']['semantic_similarity']['std']:.4f}")
    print(f"\nFull report written to: {OUT_FILE}")

    return report


if __name__ == "__main__":
    main()