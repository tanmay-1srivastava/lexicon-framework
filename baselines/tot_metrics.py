import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from rapidfuzz import fuzz


# =========================
# ✅ CONFIG (edit these)
# =========================
GT_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\data_generation\new_data\generated_datasets"
PRED_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\tot_basic_results"
GT_GLOB = "*.json"
PRED_SUFFIX = "_tot_result.json"

# Similarity threshold (0..1)
ENTITY_SIM_THRESHOLD = 0.75

# "token_set" is most robust for reordering + extra words
ENTITY_SCORE_MODE = "token_set"


# =========================
# Helpers
# =========================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_phrase(x: Any) -> str:
    return str(x).strip()

def normalize_entity(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_items(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    gtr = obj.get("ground_truth_resolutions", [])
    if not isinstance(gtr, list):
        return out

    for it in gtr:
        if not isinstance(it, dict):
            continue
        if "trigger_turn_id" not in it or "ambiguous_phrase" not in it:
            continue
        out.append(
            {
                "trigger_turn_id": int(it["trigger_turn_id"]),
                "ambiguous_phrase": normalize_phrase(it["ambiguous_phrase"]),
                "resolved_entity": normalize_entity(it.get("resolved_entity", "")),
            }
        )
    return out

def find_pred_file(pred_dir: Path, gt_file: Path, pred_suffix: str) -> Optional[Path]:
    cand = pred_dir / f"{gt_file.stem}{pred_suffix}"
    return cand if cand.exists() else None


# =========================
# Similarity scoring
# =========================
def entity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if ENTITY_SCORE_MODE == "token_set":
        return fuzz.token_set_ratio(a, b) / 100.0
    if ENTITY_SCORE_MODE == "token_sort":
        return fuzz.token_sort_ratio(a, b) / 100.0
    return fuzz.ratio(a, b) / 100.0

def entity_match(a: str, b: str) -> Tuple[bool, float]:
    sc = entity_score(a, b)
    return (sc >= ENTITY_SIM_THRESHOLD, sc)


# =========================
# Metrics (ignore TN)
# =========================
def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = 1.0 - tpr
    fpr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"accuracy": accuracy, "tpr": tpr, "fpr": fpr, "fnr": fnr, "tnr": tnr}


# =========================
# Evaluation
# =========================
def evaluate_one(gt_path: Path, pred_path: Path) -> Dict[str, Any]:
    gt_items = extract_items(load_json(gt_path))
    pred_items = extract_items(load_json(pred_path))

    gt_by_pair: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    pred_by_pair: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}

    for it in gt_items:
        gt_by_pair.setdefault((it["trigger_turn_id"], it["ambiguous_phrase"]), []).append(it)
    for it in pred_items:
        pred_by_pair.setdefault((it["trigger_turn_id"], it["ambiguous_phrase"]), []).append(it)

    gt_pairs = set(gt_by_pair.keys())
    pred_pairs = set(pred_by_pair.keys())

    # Pair-only
    pair_tp = len(gt_pairs & pred_pairs)
    pair_fp = len(pred_pairs - gt_pairs)
    pair_fn = len(gt_pairs - pred_pairs)

    # Entity fuzzy (counts entity mismatches as FP+FN)
    e_tp = e_fp = e_fn = 0
    mismatches: List[Dict[str, Any]] = []

    for pair in sorted(gt_pairs):
        gt_list = gt_by_pair.get(pair, [])
        pred_list = pred_by_pair.get(pair, [])

        if not pred_list:
            e_fn += len(gt_list)
            continue

        used = [False] * len(pred_list)

        for gt_it in gt_list:
            best_j = -1
            best_sc = -1.0
            best_pred_ent = ""

            for j, pred_it in enumerate(pred_list):
                if used[j]:
                    continue
                sc = entity_score(gt_it["resolved_entity"], pred_it["resolved_entity"])
                if sc > best_sc:
                    best_sc = sc
                    best_j = j
                    best_pred_ent = pred_it["resolved_entity"]

            if best_j == -1:
                e_fn += 1
                continue

            used[best_j] = True
            ok, sc = entity_match(gt_it["resolved_entity"], best_pred_ent)
            if ok:
                e_tp += 1
            else:
                e_fn += 1
                e_fp += 1
                mismatches.append(
                    {
                        "pair": pair,
                        "gt_entity": gt_it["resolved_entity"],
                        "pred_entity": best_pred_ent,
                        "score": sc,
                        "threshold": ENTITY_SIM_THRESHOLD,
                        "mode": ENTITY_SCORE_MODE,
                    }
                )

    # Extra preds for pairs not in GT
    for pair in sorted(pred_pairs - gt_pairs):
        e_fp += len(pred_by_pair.get(pair, []))

    return {
        "file": gt_path.name,
        "pair_match": {
            "tp": pair_tp,
            "fp": pair_fp,
            "fn": pair_fn,
            "metrics": compute_metrics(pair_tp, pair_fp, pair_fn),
        },
        "entity_fuzzy_match": {
            "tp": e_tp,
            "fp": e_fp,
            "fn": e_fn,
            "metrics": compute_metrics(e_tp, e_fp, e_fn),
            "settings": {
                "threshold": ENTITY_SIM_THRESHOLD,
                "mode": ENTITY_SCORE_MODE,
            },
            "entity_mismatches": mismatches,
        },
    }


def evaluate_batch(save_path: Optional[str] = None) -> Dict[str, Any]:
    gt_dir_p = Path(GT_DIR)
    pred_dir_p = Path(PRED_DIR)

    gt_files = sorted(gt_dir_p.glob(GT_GLOB))
    if not gt_files:
        raise RuntimeError(f"No GT files matched {GT_GLOB} in {gt_dir_p}")

    per_file: List[Dict[str, Any]] = []
    missing: List[str] = []

    p_tp = p_fp = p_fn = 0
    e_tp = e_fp = e_fn = 0

    for gt_path in gt_files:
        pred_path = find_pred_file(pred_dir_p, gt_path, PRED_SUFFIX)
        if pred_path is None:
            missing.append(gt_path.name)
            continue

        r = evaluate_one(gt_path, pred_path)
        per_file.append(r)

        p_tp += r["pair_match"]["tp"]
        p_fp += r["pair_match"]["fp"]
        p_fn += r["pair_match"]["fn"]

        e_tp += r["entity_fuzzy_match"]["tp"]
        e_fp += r["entity_fuzzy_match"]["fp"]
        e_fn += r["entity_fuzzy_match"]["fn"]

    summary = {
        "gt_dir": str(gt_dir_p),
        "pred_dir": str(pred_dir_p),
        "gt_files_total": len(gt_files),
        "evaluated_files": len(per_file),
        "missing_pred_files": len(missing),
        "missing_pred_list": missing,
        "overall": {
            "pair_match": {
                "totals": {"tp": p_tp, "fp": p_fp, "fn": p_fn},
                "metrics": compute_metrics(p_tp, p_fp, p_fn),
            },
            "entity_fuzzy_match": {
                "totals": {"tp": e_tp, "fp": e_fp, "fn": e_fn},
                "metrics": compute_metrics(e_tp, e_fp, e_fn),
                "settings": {
                    "threshold": ENTITY_SIM_THRESHOLD,
                    "mode": ENTITY_SCORE_MODE,
                },
            },
        },
        "per_file": per_file,
    }

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        with save_p.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    out_path = str(Path(PRED_DIR) / "pair_and_entity_fuzzy_metrics.json")
    metrics = evaluate_batch(save_path=out_path)

    print("PAIR OVERALL:", json.dumps(metrics["overall"]["pair_match"]["metrics"], indent=2))
    print("ENTITY OVERALL:", json.dumps(metrics["overall"]["entity_fuzzy_match"]["metrics"], indent=2))
    print("Saved:", out_path)
