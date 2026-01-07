#!/usr/bin/env python3
"""
Batch evaluation script for COT with self-reflexion results against ground truth.
Evaluates all prediction files and saves individual results to text files.
"""
import os
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers
# ----------------------------
def load_dataset(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return (text or "").lower().strip()


def match_resolutions(gt: Dict, pred: Dict, fuzzy: bool = False) -> bool:
    if normalize_text(gt.get("ambiguous_phrase")) != normalize_text(pred.get("ambiguous_phrase")):
        return False

    gt_ent = normalize_text(gt.get("resolved_entity"))
    pr_ent = normalize_text(pred.get("resolved_entity"))

    if fuzzy:
        return gt_ent == pr_ent or gt_ent in pr_ent or pr_ent in gt_ent
    return gt_ent == pr_ent


# ----------------------------
# Phrase-based evaluation
# ----------------------------
def evaluate_resolutions_cot_sr(gt_file: Path, pred_file: Path, fuzzy: bool) -> Dict[str, Any]:
    """Evaluate COT with self-reflexion output (handles list or dict format)."""
    gt = load_dataset(gt_file).get("ground_truth_resolutions", [])
    pred_raw = load_dataset(pred_file)
    
    # Handle both formats: full object or list of resolutions
    if isinstance(pred_raw, dict):
        preds = pred_raw.get("resolutions", [])
    elif isinstance(pred_raw, list):
        preds = pred_raw
    else:
        preds = []

    pred_by_phrase: Dict[str, List[Dict]] = {}
    for p in preds:
        pred_by_phrase.setdefault(normalize_text(p.get("ambiguous_phrase")), []).append(p)

    tp, fn = 0, 0
    for g in gt:
        phrase = normalize_text(g.get("ambiguous_phrase"))
        matched = any(match_resolutions(g, p, fuzzy) for p in pred_by_phrase.get(phrase, []))
        tp += int(matched)
        fn += int(not matched)

    total = len(gt)
    return {
        "tp": tp,
        "fn": fn,
        "total": total,
        "tp_rate": tp / total if total else 0.0,
        "fn_rate": fn / total if total else 0.0,
        "fuzzy": fuzzy,
    }


# ----------------------------
# Turn + phrase strict matching
# ----------------------------
def _pred_turn_1(pred: Dict) -> Optional[int]:
    if pred.get("trigger_turn_id") is not None:
        return int(pred["trigger_turn_id"])
    if pred.get("turn_index") is not None:
        return int(pred["turn_index"]) + 1
    return None


def analyze_turn_phrase_cot_sr(gt_file: Path, pred_file: Path) -> Dict[str, Any]:
    """Analyze turn + phrase matching for COT with self-reflexion output."""
    gt = load_dataset(gt_file).get("ground_truth_resolutions", [])
    pred_raw = load_dataset(pred_file)
    
    if isinstance(pred_raw, dict):
        preds = pred_raw.get("resolutions", [])
    elif isinstance(pred_raw, list):
        preds = pred_raw
    else:
        preds = []

    gt_pairs = {(g["trigger_turn_id"], normalize_text(g["ambiguous_phrase"])) for g in gt}

    match, ent_match = 0, 0
    for g in gt:
        for p in preds:
            if (_pred_turn_1(p), normalize_text(p.get("ambiguous_phrase"))) == (
                g["trigger_turn_id"],
                normalize_text(g["ambiguous_phrase"]),
            ):
                match += 1
                if normalize_text(g["resolved_entity"]) == normalize_text(p.get("resolved_entity")):
                    ent_match += 1
                break

    fp = sum(
        1
        for p in preds
        if (_pred_turn_1(p), normalize_text(p.get("ambiguous_phrase"))) not in gt_pairs
    )

    total_gt, total_pred = len(gt), len(preds)
    fn = total_gt - match

    return {
        "matching": match,
        "entity_exact_match": ent_match,
        "precision": match / total_pred if total_pred else 0.0,
        "recall": match / total_gt if total_gt else 0.0,
        "false_positive_rate": fp / total_pred if total_pred else 0.0,
        "false_negative_rate": fn / total_gt if total_gt else 0.0,
    }


# ----------------------------
# Confidence metrics
# ----------------------------
def analyze_confidence_metrics(pred_file: Path) -> Dict[str, Any]:
    """Analyze confidence and clarification metrics."""
    pred_raw = load_dataset(pred_file)
    
    if isinstance(pred_raw, dict):
        preds = pred_raw.get("resolutions", [])
    elif isinstance(pred_raw, list):
        preds = pred_raw
    else:
        preds = []

    confidences = []
    needs_clarification_count = 0
    clarifying_questions_count = 0

    for p in preds:
        conf = p.get("confidence", 0.0)
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))
        
        if p.get("needs_clarification", False):
            needs_clarification_count += 1
        
        if p.get("clarifying_question") is not None:
            clarifying_questions_count += 1

    total = len(preds)
    avg_confidence = statistics.mean(confidences) if confidences else 0.0
    median_confidence = statistics.median(confidences) if confidences else 0.0

    return {
        "total_predictions": total,
        "avg_confidence": avg_confidence,
        "median_confidence": median_confidence,
        "needs_clarification_count": needs_clarification_count,
        "needs_clarification_rate": needs_clarification_count / total if total else 0.0,
        "clarifying_questions_count": clarifying_questions_count,
    }


# ----------------------------
# File finding
# ----------------------------
def find_pred_file(gt_file: Path, pred_dir: Path, pred_suffix: str) -> Optional[Path]:
    """Maps GT filename -> prediction filename."""
    stem = gt_file.stem
    
    # Try exact match first
    candidate = pred_dir / f"{stem}{pred_suffix}.json"
    if candidate.exists():
        return candidate
    candidate2 = pred_dir / f"{stem}{pred_suffix}"
    if candidate2.exists() and candidate2.is_file():
        return candidate2
    
    # Try extracting base name (remove _user_a, _user_b, etc.)
    base_stem = re.sub(r'_user_[a-z]$', '', stem, flags=re.IGNORECASE)
    if base_stem != stem:
        candidate3 = pred_dir / f"{base_stem}{pred_suffix}.json"
        if candidate3.exists():
            return candidate3
        candidate4 = pred_dir / f"{base_stem}{pred_suffix}"
        if candidate4.exists() and candidate4.is_file():
            return candidate4
    
    return None


# ----------------------------
# Batch evaluation
# ----------------------------
def run_batch_evaluation(gt_dir: str, pred_dir: str, output_dir: str, pred_suffix: str = "_cot_self_reflexion_result", fuzzy: bool = False) -> None:
    """Run evaluation for all GT files and save results to individual txt files."""
    gt_path = Path(gt_dir).expanduser().resolve()
    pred_path = Path(pred_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all GT JSON files
    gt_files = sorted(gt_path.glob("*.json"))
    if not gt_files:
        print(f"ERROR: No JSON files found in {gt_path}")
        return
    
    print(f"Found {len(gt_files)} ground truth files")
    print(f"Output directory: {out_path}\n")
    
    results_summary = []
    missing_preds = []
    
    for i, gt_file in enumerate(gt_files, 1):
        pred_file = find_pred_file(gt_file, pred_path, pred_suffix)
        
        if pred_file is None:
            print(f"[{i}/{len(gt_files)}] ✗ SKIP {gt_file.name} - No prediction found")
            missing_preds.append(gt_file.name)
            continue
        
        output_file = out_path / f"{gt_file.stem}_eval.txt"
        json_output_file = out_path / f"{gt_file.stem}_eval.json"
        
        print(f"[{i}/{len(gt_files)}] ▶ {gt_file.name}")
        
        try:
            # Run evaluations
            eval1 = evaluate_resolutions_cot_sr(gt_file, pred_file, fuzzy)
            eval2 = analyze_turn_phrase_cot_sr(gt_file, pred_file)
            eval3 = analyze_confidence_metrics(pred_file)
            
            # Save txt output
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Evaluation for: {gt_file.name}\n")
                f.write(f"Prediction: {pred_file.name}\n")
                f.write(f"Fuzzy Matching: {fuzzy}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("PHRASE-BASED METRICS:\n")
                f.write(f"  True Positives: {eval1['tp']}/{eval1['total']}\n")
                f.write(f"  TP Rate: {eval1['tp_rate']*100:.2f}%\n")
                f.write(f"  FN Rate: {eval1['fn_rate']*100:.2f}%\n\n")
                
                f.write("TURN + PHRASE MATCHING:\n")
                f.write(f"  Matching: {eval2['matching']}\n")
                f.write(f"  Entity Exact Match: {eval2['entity_exact_match']}\n")
                f.write(f"  Precision: {eval2['precision']*100:.2f}%\n")
                f.write(f"  Recall: {eval2['recall']*100:.2f}%\n")
                f.write(f"  FP Rate: {eval2['false_positive_rate']*100:.2f}%\n")
                f.write(f"  FN Rate: {eval2['false_negative_rate']*100:.2f}%\n\n")
                
                f.write("CONFIDENCE METRICS:\n")
                f.write(f"  Total Predictions: {eval3['total_predictions']}\n")
                f.write(f"  Avg Confidence: {eval3['avg_confidence']:.3f}\n")
                f.write(f"  Median Confidence: {eval3['median_confidence']:.3f}\n")
                f.write(f"  Needs Clarification: {eval3['needs_clarification_count']}\n")
                f.write(f"  Clarification Rate: {eval3['needs_clarification_rate']*100:.2f}%\n")
            
            # Save JSON output
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "gt": str(gt_file),
                        "pred": str(pred_file),
                        "phrase_based": eval1,
                        "turn_phrase_based": eval2,
                        "confidence_metrics": eval3,
                    },
                    f,
                    indent=2,
                )
            
            # Collect for summary
            results_summary.append({
                "file": gt_file.name,
                "matching": eval2["matching"],
                "total_gt": eval1["total"],
                "recall": eval2["recall"],
                "entity_match": eval2["entity_exact_match"],
                "precision": eval2["precision"],
                "avg_confidence": eval3["avg_confidence"],
                "clarification_rate": eval3["needs_clarification_rate"],
            })
            
            print(f"  → Saved to {output_file.name}")
        
        except Exception as e:
            print(f"  ✗ ERROR for {gt_file.name}: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(gt_files)}")
    print(f"Successfully evaluated: {len(results_summary)}")
    print(f"Missing predictions: {len(missing_preds)}")
    
    if results_summary:
        print("\nPer-file Results:")
        print("-" * 100)
        print(f"{'File':<40} {'Recall %':>10} {'Entity Match':>15} {'Avg Conf':>12}")
        print("-" * 100)
        for r in results_summary:
            print(f"{r['file']:<40} {r['recall']*100:>9.1f}% {r['entity_match']:>15} {r['avg_confidence']:>12.3f}")
        
        avg_recall = sum(r['recall'] for r in results_summary) / len(results_summary)
        avg_entity = sum(r['entity_match'] for r in results_summary) / len(results_summary)
        avg_conf = sum(r['avg_confidence'] for r in results_summary) / len(results_summary)
        print("-" * 100)
        print(f"{'AVERAGE':<40} {avg_recall*100:>9.1f}% {avg_entity:>15.1f} {avg_conf:>12.3f}")
        print("=" * 80)
    
    if missing_preds:
        print("\nMissing Predictions for:")
        for f in missing_preds:
            print(f"  - {f}")
    
    # Save summary to JSON
    summary_file = out_path / "batch_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_files": len(gt_files),
            "evaluated": len(results_summary),
            "missing": len(missing_preds),
            "results": results_summary,
            "missing_files": missing_preds,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluate COT with self-reflexion predictions")
    parser.add_argument(
        "--gt-dir",
        default="/Users/amartya/Documents/lexicon-framework/data_generation/new_data/generated_datasets",
        help="Ground truth directory"
    )
    parser.add_argument(
        "--pred-dir",
        default="/Users/amartya/Documents/lexicon-framework/baselines",
        help="Prediction directory"
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/amartya/Documents/lexicon-framework/baselines/batch_evals",
        help="Output directory for results"
    )
    parser.add_argument(
        "--pred-suffix",
        default="_cot_self_reflexion_result",
        help="Prediction file suffix"
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Use fuzzy entity matching"
    )
    
    args = parser.parse_args()
    
    run_batch_evaluation(
        args.gt_dir,
        args.pred_dir,
        args.out_dir,
        args.pred_suffix,
        args.fuzzy
    )
