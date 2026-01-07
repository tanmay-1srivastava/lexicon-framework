#!/usr/bin/env python3
"""
Batch evaluation script that runs cot_base.py evaluation for all JSON files
and saves individual results to text files.
"""
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def find_pred_file(gt_file: Path, pred_dir: Path, pred_suffix: str) -> Optional[Path]:
    """
    Maps GT filename -> prediction filename.
    """
    stem = gt_file.stem
    
    # Try exact match first
    candidate = pred_dir / f"{stem}{pred_suffix}.json"
    if candidate.exists():
        return candidate
    candidate2 = pred_dir / f"{stem}{pred_suffix}"
    if candidate2.exists() and candidate2.is_file():
        return candidate2
    
    # Try extracting base name (remove _user_a, _user_b, etc.)
    import re
    base_stem = re.sub(r'_user_[a-z]$', '', stem, flags=re.IGNORECASE)
    if base_stem != stem:
        candidate3 = pred_dir / f"{base_stem}{pred_suffix}.json"
        if candidate3.exists():
            return candidate3
        candidate4 = pred_dir / f"{base_stem}{pred_suffix}"
        if candidate4.exists() and candidate4.is_file():
            return candidate4
    
    return None


def run_batch_evaluation(gt_dir: str, pred_dir: str, output_dir: str, pred_suffix: str = "_cot_result", fuzzy: bool = False) -> None:
    """
    Run evaluation for all GT files and save results to individual txt files.
    """
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
        
        print(f"[{i}/{len(gt_files)}] ▶ {gt_file.name}")
        
        # Run evaluation
        cmd = [
            "/Users/amartya/Documents/lexicon-framework/lexicon/bin/python",
            "baselines/cot_base.py",
            "--evaluate",
            "--ground-truth", str(gt_file),
            "--predictions", str(pred_file),
            "--out", str(out_path / f"{gt_file.stem}_eval.json")
        ]
        
        if fuzzy:
            cmd.append("--fuzzy")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Save output to txt file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Evaluation for: {gt_file.name}\n")
                f.write(f"Prediction: {pred_file.name}\n")
                f.write(f"Fuzzy Matching: {fuzzy}\n")
                f.write("=" * 80 + "\n\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
            
            # Parse JSON output to extract metrics
            json_file = out_path / f"{gt_file.stem}_eval.json"
            if json_file.exists():
                with open(json_file, "r") as jf:
                    eval_data = json.load(jf)
                    turn_phrase = eval_data.get("turn_phrase_based", {})
                    results_summary.append({
                        "file": gt_file.name,
                        "matching": turn_phrase.get("matching", 0),
                        "total_gt": turn_phrase.get("total_ground_truth", 0),
                        "matching_pct": turn_phrase.get("recall", 0),
                        "entity_match": turn_phrase.get("entity_exact_match", 0),
                        "precision": turn_phrase.get("precision", 0),
                    })
            
            print(f"  → Saved to {output_file.name}")
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT for {gt_file.name}")
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
        print("-" * 80)
        print(f"{'File':<40} {'Match %':>10} {'Entity Match':>15}")
        print("-" * 80)
        for r in results_summary:
            print(f"{r['file']:<40} {r['matching_pct']:>9.1f}% {r['entity_match']:>15}")
        
        avg_matching = sum(r['matching_pct'] for r in results_summary) / len(results_summary)
        avg_entity = sum(r['entity_match'] for r in results_summary) / len(results_summary)
        print("-" * 80)
        print(f"{'AVERAGE':<40} {avg_matching:>9.1f}% {avg_entity:>15.1f}")
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
    
    parser = argparse.ArgumentParser(description="Batch evaluate all COT predictions")
    parser.add_argument(
        "--gt-dir",
        default="/Users/amartya/Documents/lexicon-framework/data_generation/new_data/generated_datasets",
        help="Ground truth directory"
    )
    parser.add_argument(
        "--pred-dir",
        default="/Users/amartya/Documents/lexicon-framework/baselines/generated_datasets",
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
