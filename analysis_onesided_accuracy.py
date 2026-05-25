

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util


# =========================
# CONFIG
# =========================
DATASETS_DIR = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\data_generation\event_keywords\generated_datasets"
PRED_DIR     = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\single_turn_predictions"
OUT_FILE     = r"C:\Users\amart\Desktop\PhD projects\lexicon-framework\baselines\single_turn_predictions\eval_resolution_report.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Suffixes used by the GPT prediction batch script
SUFFIX_USER_B = "_single_turn_predictions_userB.json"
SUFFIX_USER_A = "_single_turn_predictions_userA.json"

# When to count a row as a successful resolution.
# A row is a HIT if any of these conditions holds:
#   - substring match (case-insensitive) of resolved_entity in predicted_text
#   - OR token F1 >= HIT_THRESHOLD_F1
#   - OR semantic similarity >= HIT_THRESHOLD_SEM
HIT_THRESHOLD_F1 = 0.5
HIT_THRESHOLD_SEM = 0.65

# Tokens to strip when normalizing
STRIP_CHARS = ' \t\n.,;:!?"\'`()[]{}—–-*'


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
# Text helpers
# =========================
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("*", "")
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def substring_match(predicted: str, resolved: str) -> int:
    """1 if resolved_entity (normalized) appears in predicted (normalized) as a substring."""
    p = normalize_text(predicted)
    r = normalize_text(resolved)
    if not r or not p:
        return 0
    return 1 if r in p else 0


def token_f1(a: str, b: str) -> float:
    """Set-based F1 over tokens. 0 if either is empty."""
    aa = set(normalize_text(a).split())
    bb = set(normalize_text(b).split())
    # Drop common stopwords-ish noise to make small entities not dominated by 'the', 'a'
    drop = {"the", "a", "an", "of", "to", "for", "in", "on", "at"}
    aa = {x for x in aa if x not in drop}
    bb = {x for x in bb if x not in drop}
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
# Build (turn_id -> resolutions list) for a dataset file
# =========================
def group_resolutions_by_turn(dataset_obj: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in dataset_obj.get("ground_truth_resolutions", []) or []:
        if not isinstance(r, dict):
            continue
        try:
            tid = int(r.get("trigger_turn_id"))
        except Exception:
            continue
        out.setdefault(tid, []).append(r)
    return out


# =========================
# Per-pass evaluation
# =========================
def evaluate_pass(
    datasets_dir: Path,
    pred_dir: Path,
    suffix: str,
    pass_name: str,
    model: SentenceTransformer,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Iterate over all prediction files matching the given suffix, score each row,
    and return (rows, aggregate_report).
    """
    pred_files = sorted(pred_dir.glob(f"*{suffix}"))
    if not pred_files:
        return [], {
            "pass": pass_name,
            "suffix": suffix,
            "num_files": 0,
            "n_rows": 0,
            "hit_rate": float("nan"),
            "token_f1": aggregate(np.array([])),
            "semantic_similarity": aggregate(np.array([])),
            "per_file": [],
        }

    all_rows: List[Dict[str, Any]] = []
    per_file_records: List[Dict[str, Any]] = []

    for pred_path in pred_files:
        # Map pred file -> dataset file
        stem = pred_path.name[: -len(suffix)]
        dataset_path = datasets_dir / (stem + ".json")
        if not dataset_path.exists():
            print(f"[{pass_name}] skip: dataset not found for {pred_path.name}")
            continue

        try:
            pred_doc = load_json(pred_path)
            dataset = load_json(dataset_path)
        except Exception as e:
            print(f"[{pass_name}] skip: failed to load {pred_path.name}: {e}")
            continue

        res_by_turn = group_resolutions_by_turn(dataset)
        predictions = pred_doc.get("predictions", []) or []

        file_rows: List[Dict[str, Any]] = []

        for pred in predictions:
            tid = pred.get("masked_turn_id")
            try:
                tid = int(tid)
            except Exception:
                continue
            predicted_text = pred.get("predicted_text", "") or ""
            ground_truth_text = pred.get("ground_truth_text", "") or ""

            rez_list = res_by_turn.get(tid, [])
            if not rez_list:
                # No ambiguous phrases to resolve for this turn — skip it from
                # the resolution-accuracy metric.
                continue

            for r in rez_list:
                amb = r.get("ambiguous_phrase", "") or ""
                entity = r.get("resolved_entity", "") or ""
                source = r.get("resolution_source", "") or ""

                if not entity:
                    continue

                sub = substring_match(predicted_text, entity)
                f1 = token_f1(predicted_text, entity)

                file_rows.append({
                    "file": pred_doc.get("file", pred_path.name),
                    "turn_id": tid,
                    "masked_speaker": pred.get("masked_speaker", ""),
                    "ambiguous_phrase": amb,
                    "resolved_entity": entity,
                    "resolution_source": source,
                    "ground_truth_text": ground_truth_text,
                    "predicted_text": predicted_text,
                    "substring_match": sub,
                    "token_f1": f1,
                    # placeholder, filled below in batch
                    "semantic_similarity": 0.0,
                    "is_hit": 0,
                })

        # Batched semantic similarity for this file's rows
        if file_rows:
            preds = [r["predicted_text"] for r in file_rows]
            ents  = [r["resolved_entity"] for r in file_rows]
            emb_p = model.encode(preds, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            emb_e = model.encode(ents,  convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            sim = util.cos_sim(emb_p, emb_e).cpu().numpy()
            sim_diag = np.diag(sim)
            for i, row in enumerate(file_rows):
                row["semantic_similarity"] = float(sim_diag[i])
                # Decide hit
                row["is_hit"] = int(
                    row["substring_match"] == 1
                    or row["token_f1"] >= HIT_THRESHOLD_F1
                    or row["semantic_similarity"] >= HIT_THRESHOLD_SEM
                )

        all_rows.extend(file_rows)

        # Per-file aggregate
        if file_rows:
            arr_f1 = np.array([r["token_f1"] for r in file_rows])
            arr_sim = np.array([r["semantic_similarity"] for r in file_rows])
            n_hits = sum(r["is_hit"] for r in file_rows)
            n_sub = sum(r["substring_match"] for r in file_rows)
            per_file_records.append({
                "file": pred_doc.get("file", pred_path.name),
                "n_rows": len(file_rows),
                "hit_rate": n_hits / len(file_rows),
                "substring_hit_rate": n_sub / len(file_rows),
                "token_f1": aggregate(arr_f1),
                "semantic_similarity": aggregate(arr_sim),
            })
        else:
            per_file_records.append({
                "file": pred_doc.get("file", pred_path.name),
                "n_rows": 0,
                "hit_rate": float("nan"),
                "substring_hit_rate": float("nan"),
                "token_f1": aggregate(np.array([])),
                "semantic_similarity": aggregate(np.array([])),
            })

        print(f"[{pass_name}] {pred_path.name} -> {len(file_rows)} rows")

    # Global aggregate for this pass
    if all_rows:
        arr_f1 = np.array([r["token_f1"] for r in all_rows])
        arr_sim = np.array([r["semantic_similarity"] for r in all_rows])
        n_hits = sum(r["is_hit"] for r in all_rows)
        n_sub = sum(r["substring_match"] for r in all_rows)
        pass_report = {
            "pass": pass_name,
            "suffix": suffix,
            "num_files": len(pred_files),
            "n_rows": len(all_rows),
            "hit_count": int(n_hits),
            "hit_rate": n_hits / len(all_rows),
            "substring_hit_count": int(n_sub),
            "substring_hit_rate": n_sub / len(all_rows),
            "token_f1": aggregate(arr_f1),
            "semantic_similarity": aggregate(arr_sim),
            "per_file": per_file_records,
        }
    else:
        pass_report = {
            "pass": pass_name,
            "suffix": suffix,
            "num_files": len(pred_files),
            "n_rows": 0,
            "hit_count": 0,
            "hit_rate": float("nan"),
            "substring_hit_count": 0,
            "substring_hit_rate": float("nan"),
            "token_f1": aggregate(np.array([])),
            "semantic_similarity": aggregate(np.array([])),
            "per_file": per_file_records,
        }
    return all_rows, pass_report


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

    rows_b, report_b = evaluate_pass(
        datasets_dir=datasets_dir,
        pred_dir=pred_dir,
        suffix=SUFFIX_USER_B,
        pass_name="User B masked",
        model=model,
    )
    rows_a, report_a = evaluate_pass(
        datasets_dir=datasets_dir,
        pred_dir=pred_dir,
        suffix=SUFFIX_USER_A,
        pass_name="User A masked",
        model=model,
    )

    report = {
        "datasets_dir": str(datasets_dir),
        "pred_dir": str(pred_dir),
        "embedding_model": EMBED_MODEL_NAME,
        "hit_definition": {
            "substring_match": "resolved_entity appears (case-insensitive) in predicted_text",
            "token_f1_threshold": HIT_THRESHOLD_F1,
            "semantic_similarity_threshold": HIT_THRESHOLD_SEM,
            "rule": "row is a HIT if ANY of the three conditions is true",
        },
        "passes": [report_b, report_a],
        "rows": {"user_b_masked": rows_b, "user_a_masked": rows_a},
    }

    save_json(Path(OUT_FILE), report)

    # ---- Pretty print ----
    print("\n================ RESOLUTION-ACCURACY SUMMARY ================")
    for r in (report_b, report_a):
        print(f"\n--- {r['pass']} ---")
        print(f"  Files: {r['num_files']}, Rows scored: {r['n_rows']}")
        if r["n_rows"]:
            print(f"  HITS (any criterion)    : {r['hit_count']} / {r['n_rows']}  =  {r['hit_rate']*100:.2f}%")
            print(f"  Substring hits          : {r['substring_hit_count']} / {r['n_rows']}  =  {r['substring_hit_rate']*100:.2f}%")
            print(f"  Token F1                : mean={r['token_f1']['mean']:.4f}  "
                  f"median={r['token_f1']['median']:.4f}  std={r['token_f1']['std']:.4f}")
            print(f"  Semantic similarity     : mean={r['semantic_similarity']['mean']:.4f}  "
                  f"median={r['semantic_similarity']['median']:.4f}  std={r['semantic_similarity']['std']:.4f}")
    print(f"\nFull report written to: {OUT_FILE}")

    return report


if __name__ == "__main__":
    main()