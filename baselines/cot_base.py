#!/usr/bin/env python3
import json
import os
import argparse
import sys
from typing import Dict, Any, List, Optional


data_dir = "/Users/amartya/Documents/lexicon-framework/data_generation/new_data/generated_datasets"
file_name = "doctor_visit_001.json"


def load_dataset(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_resolutions(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    resolutions = dataset.get("ground_truth_resolutions", [])
    extracted = []
    for r in resolutions:
        ambiguous = r.get("ambiguous_phrase")
        resolved = r.get("resolved_entity")
        if ambiguous is None and resolved is None:
            continue
        extracted.append({
            "trigger_turn_id": r.get("trigger_turn_id"),
            "ambiguous_phrase": ambiguous,
            "resolved_entity": resolved,
            "resolution_source": r.get("resolution_source"),
        })
    return extracted


def normalize_text(text: str) -> str:
    return (text or "").lower().strip()


def match_resolutions(ground_truth: Dict[str, Any], prediction: Dict[str, Any], fuzzy: bool = False) -> bool:
    gt_phrase = normalize_text(ground_truth.get("ambiguous_phrase", ""))
    pred_phrase = normalize_text(prediction.get("ambiguous_phrase", ""))
    if gt_phrase != pred_phrase:
        return False

    gt_entity = normalize_text(ground_truth.get("resolved_entity", ""))
    pred_entity = normalize_text(prediction.get("resolved_entity", ""))

    if fuzzy:
        return (gt_entity in pred_entity) or (pred_entity in gt_entity) or (gt_entity == pred_entity)
    return gt_entity == pred_entity


def _pred_turn_id_1_indexed(pred: Dict[str, Any]) -> Optional[int]:
    """
    Return predicted turn id in 1-indexed form if available.
    - If pred has trigger_turn_id: assume it's already 1-indexed (your COT schema).
    - Else if pred has turn_index: convert 0-index -> 1-index.
    """
    if pred.get("trigger_turn_id") is not None:
        try:
            return int(pred["trigger_turn_id"])
        except Exception:
            return None
    if pred.get("turn_index") is not None:
        try:
            return int(pred["turn_index"]) + 1
        except Exception:
            return None
    return None


def evaluate_resolutions(ground_truth_file: str, prediction_file: str, fuzzy: bool = False, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print(f"Loading ground truth from: {ground_truth_file}")
    dataset = load_dataset(ground_truth_file)
    ground_truth_resolutions = dataset.get("ground_truth_resolutions", [])

    if verbose:
        print(f"Loading predictions from: {prediction_file}")
    predictions = load_dataset(prediction_file)
    if not isinstance(predictions, list):
        predictions = [predictions]

    if verbose:
        print(f"\nGround truth resolutions: {len(ground_truth_resolutions)}")
        print(f"Model predictions: {len(predictions)}")

    pred_by_phrase: Dict[str, List[Dict[str, Any]]] = {}
    for pred in predictions:
        phrase = normalize_text(pred.get("ambiguous_phrase", ""))
        pred_by_phrase.setdefault(phrase, []).append(pred)

    true_positives = 0
    false_negatives = 0

    if verbose:
        print("\n" + "=" * 80)
        print("DETAILED MATCHING RESULTS (Matching by Ambiguous Phrase):")
        print("=" * 80)

    for i, gt in enumerate(ground_truth_resolutions):
        gt_turn = gt.get("trigger_turn_id")  # 1-indexed
        gt_phrase = normalize_text(gt.get("ambiguous_phrase", ""))
        gt_entity = normalize_text(gt.get("resolved_entity", ""))

        found_match = False
        if gt_phrase in pred_by_phrase:
            for pred in pred_by_phrase[gt_phrase]:
                if match_resolutions(gt, pred, fuzzy=fuzzy):
                    true_positives += 1
                    found_match = True

                    if verbose:
                        pred_turn_1 = _pred_turn_id_1_indexed(pred)
                        turn_match = "✓" if (pred_turn_1 is not None and pred_turn_1 == gt_turn) else "✗"
                        print(f"\n✓ TRUE POSITIVE (GT #{i}):")
                        print(f"  Phrase: '{gt_phrase}'")
                        print(f"  Turn ID: GT={gt_turn}, Pred={pred_turn_1} {turn_match}")
                        print(f"  Ground Truth Entity:  {gt_entity}")
                        print(f"  Prediction Entity:    {normalize_text(pred.get('resolved_entity', ''))}")
                    break

        if not found_match:
            false_negatives += 1
            if verbose:
                print(f"\n✗ FALSE NEGATIVE (GT #{i}):")
                print(f"  Phrase: '{gt_phrase}'")
                print(f"  Turn ID: {gt_turn}")
                print(f"  Expected: {gt_entity}")
                if gt_phrase in pred_by_phrase:
                    actual = normalize_text(pred_by_phrase[gt_phrase][0].get("resolved_entity", ""))
                    pred_turn_1 = _pred_turn_id_1_indexed(pred_by_phrase[gt_phrase][0])
                    print(f"  Got:      {actual} (Pred Turn ID: {pred_turn_1})")
                else:
                    print("  Got:      (no prediction found for this phrase)")

    total_ground_truth = len(ground_truth_resolutions)
    tp_rate = (true_positives / total_ground_truth * 100) if total_ground_truth > 0 else 0.0
    fn_rate = (false_negatives / total_ground_truth * 100) if total_ground_truth > 0 else 0.0

    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY:")
        print("=" * 80)
        print(f"Ground Truth Resolutions:    {total_ground_truth}")
        print(f"True Positives (TP):         {true_positives}")
        print(f"False Negatives (FN):        {false_negatives}")
        print(f"\nTrue Positive Rate (TPR):    {tp_rate:.2f}%")
        print(f"False Negative Rate (FNR):   {fn_rate:.2f}%")
        print("=" * 80)

    return {
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "total_ground_truth": total_ground_truth,
        "tp_rate": tp_rate,
        "fn_rate": fn_rate,
        "fuzzy_matching": fuzzy,
    }


def analyze_turn_id_phrase_matching(ground_truth_file: str, prediction_file: str, verbose: bool = True) -> Dict[str, Any]:
    dataset = load_dataset(ground_truth_file)
    ground_truth = dataset.get("ground_truth_resolutions", [])

    predictions = load_dataset(prediction_file)
    if not isinstance(predictions, list):
        predictions = [predictions]

    matching_count = 0
    entity_match_count = 0

    # Build GT pairs
    gt_pairs = []
    for gt in ground_truth:
        gt_turn_1 = gt.get("trigger_turn_id")
        gt_phrase = normalize_text(gt.get("ambiguous_phrase", ""))
        gt_pairs.append((gt_turn_1, gt_phrase, normalize_text(gt.get("resolved_entity", ""))))

    # Build prediction phrase-only map for flexible matching
    pred_by_phrase = {}
    for pred in predictions:
        pred_phrase = normalize_text(pred.get("ambiguous_phrase", ""))
        if pred_phrase not in pred_by_phrase:
            pred_by_phrase[pred_phrase] = pred

    # Count phrase matches (ignoring turn_id)
    matched_phrases = set()
    for (gt_turn_1, gt_phrase, gt_ent) in gt_pairs:
        if gt_phrase in pred_by_phrase:
            pred = pred_by_phrase[gt_phrase]
            matching_count += 1
            matched_phrases.add(gt_phrase)
            if gt_ent == normalize_text(pred.get("resolved_entity", "")):
                entity_match_count += 1

    # False positives: phrases in predictions not in GT
    gt_phrase_set = {p for (_, p, _) in gt_pairs}
    false_positive_count = len(pred_by_phrase) - len(matched_phrases)

    total_gt = len(ground_truth)
    total_pred = len(pred_by_phrase)
    not_found = total_gt - matching_count

    precision = (matching_count / total_pred * 100) if total_pred > 0 else 0.0
    recall = (matching_count / total_gt * 100) if total_gt > 0 else 0.0
    accuracy = recall
    false_negative_rate = (not_found / total_gt * 100) if total_gt > 0 else 0.0
    false_positive_rate = (false_positive_count / total_pred * 100) if total_pred > 0 else 0.0

    if verbose:
        print("\n" + "=" * 80)
        print("PHRASE MATCHING ANALYSIS (Ignoring Turn ID)")
        print("=" * 80)
        print(f"Total Ground Truth Items:           {total_gt}")
        print(f"Total Unique Prediction Phrases:    {total_pred}")
        print(f"\nMatching Phrases:                   {matching_count} / {total_gt} ({100*matching_count/total_gt:.1f}%)")
        print(f"Not Found (False Negatives):        {not_found} / {total_gt} ({false_negative_rate:.1f}%)")
        print(f"Extra Phrases (False Pos):          {false_positive_count} / {total_pred} ({false_positive_rate:.1f}%)")
        print(f"\nEntity Exact Match (for matches):   {entity_match_count} / {matching_count} ({100*entity_match_count/matching_count if matching_count > 0 else 0:.1f}%)")
        print(f"\nPrecision:                          {precision:.1f}%")
        print(f"Recall:                             {recall:.1f}%")
        print(f"Accuracy:                           {accuracy:.1f}%")
        print("=" * 80)

    return {
        "total_ground_truth": total_gt,
        "total_predictions": total_pred,
        "matching": matching_count,
        "false_negatives": not_found,
        "false_positives": false_positive_count,
        "entity_exact_match": entity_match_count,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract GT resolutions or evaluate predictions")
    parser.add_argument("--file", "-f", default=file_name, help="Dataset filename in data_dir")
    parser.add_argument("--dir", "-d", default=data_dir, help="Data directory")
    parser.add_argument("--out", "-o", default=None, help="Output JSON file (optional)")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Evaluate mode")
    parser.add_argument("--ground-truth", "-g", default=None, help="Path to ground truth JSON file")
    parser.add_argument("--predictions", "-p", default=None, help="Path to model predictions JSON file")
    parser.add_argument("--fuzzy", action="store_true", help="Use fuzzy matching for resolved entities")
    parser.add_argument("--quiet", action="store_true", help="Less printing (recommended for batch)")
    args = parser.parse_args(argv)

    verbose = not args.quiet

    if args.evaluate:
        if not args.ground_truth or not args.predictions:
            print("Error: --ground-truth and --predictions are required for evaluation mode", file=sys.stderr)
            return 1

        # Run both analyses and write combined output if requested
        eval_phrase = evaluate_resolutions(args.ground_truth, args.predictions, fuzzy=args.fuzzy, verbose=verbose)
        eval_turn_phrase = analyze_turn_id_phrase_matching(args.ground_truth, args.predictions, verbose=verbose)

        combined = {
            "ground_truth": args.ground_truth,
            "predictions": args.predictions,
            "phrase_based": eval_phrase,
            "turn_phrase_based": eval_turn_phrase,
        }

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2)
                f.write("\n")
            if verbose:
                print(f"\nResults saved to: {args.out}")
        else:
            print(json.dumps(combined, indent=2))

        return 0

    # Extraction mode
    file_path = os.path.join(args.dir, args.file)
    if verbose:
        print(f"Loading data from {file_path}")
    try:
        dataset = load_dataset(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 2

    extracted = extract_resolutions(dataset)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fo:
                json.dump(extracted, fo, ensure_ascii=False, indent=2)
                fo.write("\n")
            if verbose:
                print(f"Wrote {len(extracted)} resolutions to {args.out}")
        except Exception as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            return 3
    else:
        print(json.dumps(extracted, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
