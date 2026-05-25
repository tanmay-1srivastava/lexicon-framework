"""
Compute semantic similarity and sentence similarity between predicted and
ground-truth turns, separately for the User B masking pass and the User A
masking pass.

Inputs (from the previous batch script):
  - <stem>_single_turn_predictions_userB.json
  - <stem>_single_turn_predictions_userA.json

For each prediction:
  - semantic_similarity = cosine similarity of Sentence-BERT embeddings
                          (predicted_text vs ground_truth_text)
  - sentence_similarity = ROUGE-L F1 (lexical overlap)

For each pass (User B, User A) we report mean / median / std for both metrics.

Dependencies (install once):
    pip install sentence-transformers rouge-score numpy
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer


# =========================
# CONFIG
# =========================
# Directory where the previous script wrote its outputs
PRED_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\single_turn_predictions"

# Suffixes used by the batch prediction script
SUFFIX_USER_B = "_single_turn_predictions_userB.json"
SUFFIX_USER_A = "_single_turn_predictions_userA.json"

# Where to write the evaluation report
OUT_FILE = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\single_turn_predictions\similarity_report.json"

# Sentence-BERT model for semantic similarity.
# 'all-MiniLM-L6-v2' is fast and a strong default.
# Use 'all-mpnet-base-v2' for higher quality (slower).
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Skip predictions whose ground truth or prediction is empty after cleaning
SKIP_EMPTY = True


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
# Text cleaning
# =========================
def clean_text(s: str) -> str:
    """
    Light normalization: strip surrounding quotes/whitespace, collapse spaces.
    We DO NOT lowercase here because Sentence-BERT handles case fine and we
    want ROUGE to see the model's casing too.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # Strip wrapping quotes/backticks if the model added any
    s = s.strip('"').strip("'").strip("`").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# Collect (pred, gt) pairs from a single prediction file
# =========================
def collect_pairs(pred_file: Path) -> List[Tuple[str, str, int]]:
    """
    Returns list of (predicted_text, ground_truth_text, masked_turn_id).
    """
    data = load_json(pred_file)
    preds = data.get("predictions", []) or []
    out = []
    for p in preds:
        pred = clean_text(p.get("predicted_text", ""))
        gt = clean_text(p.get("ground_truth_text", ""))
        tid = p.get("masked_turn_id")
        if SKIP_EMPTY and (not pred or not gt):
            continue
        out.append((pred, gt, tid))
    return out


# =========================
# Metrics
# =========================
def semantic_similarities(
    model: SentenceTransformer,
    preds: List[str],
    gts: List[str],
) -> np.ndarray:
    """
    Returns a 1-D np.array of cosine similarities, one per (pred, gt) pair.
    """
    if not preds:
        return np.array([], dtype=float)

    emb_pred = model.encode(preds, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    emb_gt = model.encode(gts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    # Pairwise diagonal: util.cos_sim returns NxN; we only want the diagonal
    sim_matrix = util.cos_sim(emb_pred, emb_gt).cpu().numpy()
    diag = np.diag(sim_matrix)
    return diag.astype(float)


def rouge_l_f1_scores(
    scorer: rouge_scorer.RougeScorer,
    preds: List[str],
    gts: List[str],
) -> np.ndarray:
    """
    Returns 1-D np.array of ROUGE-L F1 scores, one per pair.
    """
    out = []
    for p, g in zip(preds, gts):
        if not p or not g:
            out.append(0.0)
            continue
        score = scorer.score(g, p)  # (target, prediction)
        out.append(score["rougeL"].fmeasure)
    return np.array(out, dtype=float)


# =========================
# Aggregation
# =========================
def aggregate(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),  # population std; switch to ddof=1 for sample std
    }


# =========================
# Per-pass evaluation
# =========================
def evaluate_pass(
    pred_dir: Path,
    suffix: str,
    pass_name: str,
    model: SentenceTransformer,
    scorer: rouge_scorer.RougeScorer,
) -> Dict[str, Any]:
    files = sorted(pred_dir.glob(f"*{suffix}"))
    if not files:
        print(f"[{pass_name}] No files found matching *{suffix} in {pred_dir}")
        return {
            "pass": pass_name,
            "suffix": suffix,
            "num_files": 0,
            "num_predictions": 0,
            "semantic_similarity": aggregate(np.array([])),
            "sentence_similarity_rougeL_f1": aggregate(np.array([])),
            "per_file": [],
        }

    all_preds: List[str] = []
    all_gts: List[str] = []
    per_file_records: List[Dict[str, Any]] = []
    pair_origin: List[str] = []  # which file each pair came from (for per-file stats)

    for fp in files:
        pairs = collect_pairs(fp)
        if not pairs:
            per_file_records.append({
                "file": fp.name,
                "num_predictions": 0,
                "semantic_similarity": aggregate(np.array([])),
                "sentence_similarity_rougeL_f1": aggregate(np.array([])),
            })
            continue
        preds = [p for (p, g, _t) in pairs]
        gts = [g for (p, g, _t) in pairs]
        all_preds.extend(preds)
        all_gts.extend(gts)
        pair_origin.extend([fp.name] * len(pairs))
        # Placeholder; per-file stats filled in after global scoring (more efficient)
        per_file_records.append({"file": fp.name, "num_predictions": len(pairs)})

    # Score everything in one go (fast: single batched encode)
    print(f"[{pass_name}] Encoding {len(all_preds)} prediction/gt pairs...")
    sem = semantic_similarities(model, all_preds, all_gts)
    rouge = rouge_l_f1_scores(scorer, all_preds, all_gts)

    # Fill per-file aggregates by slicing
    cursor = 0
    for rec in per_file_records:
        n = rec.get("num_predictions", 0)
        if n == 0:
            rec["semantic_similarity"] = aggregate(np.array([]))
            rec["sentence_similarity_rougeL_f1"] = aggregate(np.array([]))
            continue
        sem_slice = sem[cursor:cursor + n]
        rouge_slice = rouge[cursor:cursor + n]
        rec["semantic_similarity"] = aggregate(sem_slice)
        rec["sentence_similarity_rougeL_f1"] = aggregate(rouge_slice)
        cursor += n

    return {
        "pass": pass_name,
        "suffix": suffix,
        "num_files": len(files),
        "num_predictions": int(len(all_preds)),
        "semantic_similarity": aggregate(sem),
        "sentence_similarity_rougeL_f1": aggregate(rouge),
        "per_file": per_file_records,
    }


# =========================
# Main
# =========================
def main() -> Dict[str, Any]:
    pred_dir = Path(PRED_DIR)
    if not pred_dir.exists():
        raise RuntimeError(f"PRED_DIR does not exist: {pred_dir}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    report: Dict[str, Any] = {
        "pred_dir": str(pred_dir),
        "embedding_model": EMBED_MODEL_NAME,
        "metrics": {
            "semantic_similarity": "Cosine similarity of Sentence-BERT embeddings (predicted vs ground truth).",
            "sentence_similarity_rougeL_f1": "ROUGE-L F1 score (lexical sentence-level overlap).",
        },
        "passes": [],
    }

    # ---- User B pass (masked first per your spec) ----
    pass_b = evaluate_pass(
        pred_dir=pred_dir,
        suffix=SUFFIX_USER_B,
        pass_name="User B masked",
        model=model,
        scorer=scorer,
    )
    report["passes"].append(pass_b)

    # ---- User A pass (masked second) ----
    pass_a = evaluate_pass(
        pred_dir=pred_dir,
        suffix=SUFFIX_USER_A,
        pass_name="User A masked",
        model=model,
        scorer=scorer,
    )
    report["passes"].append(pass_a)

    # ---- Save and pretty-print ----
    save_json(Path(OUT_FILE), report)

    print("\n================ SUMMARY ================")
    for p in report["passes"]:
        print(f"\n--- {p['pass']} ---")
        print(f"  files: {p['num_files']},  predictions: {p['num_predictions']}")
        sem = p["semantic_similarity"]
        rouge = p["sentence_similarity_rougeL_f1"]
        print(f"  Semantic similarity  : mean={sem['mean']:.4f}  median={sem['median']:.4f}  std={sem['std']:.4f}")
        print(f"  Sentence sim (RougeL): mean={rouge['mean']:.4f}  median={rouge['median']:.4f}  std={rouge['std']:.4f}")
    print(f"\nFull report written to: {OUT_FILE}")

    return report


if __name__ == "__main__":
    main()