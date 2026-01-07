#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers (unchanged logic)
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
def evaluate_resolutions(gt_file: Path, pred_file: Path, fuzzy: bool) -> Dict[str, Any]:
    gt = load_dataset(gt_file).get("ground_truth_resolutions", [])
    preds = load_dataset(pred_file)
    if not isinstance(preds, list):
        preds = [preds]

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


def analyze_turn_phrase(gt_file: Path, pred_file: Path) -> Dict[str, Any]:
    gt = load_dataset(gt_file).get("ground_truth_resolutions", [])
    preds = load_dataset(pred_file)
    if not isinstance(preds, list):
        preds = [preds]

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
# Batch logic
# ----------------------------
def find_pred(gt: Path, pred_dir: Path, suffix: str) -> Optional[Path]:
    stem = gt.stem
    base = re.sub(r"_user_[a-z]$", "", stem, flags=re.I)
    for name in (
        f"{stem}{suffix}.json",
        f"{base}{suffix}.json",
    ):
        p = pred_dir / name
        if p.exists():
            return p
    return None


def summarize(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    return statistics.median(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pred-suffix", default="_cot_result")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--fuzzy", action="store_true")
    p.add_argument("--skip-missing", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.rglob("*.json") if args.recursive else gt_dir.glob("*.json"))

    tpr, fnr, prec, rec = [], [], [], []
    evaluated, missing = 0, 0

    for i, gt in enumerate(gt_files, 1):
        pred = find_pred(gt, pred_dir, args.pred_suffix)
        if pred is None:
            missing += 1
            if not args.skip_missing:
                print(f"[{i}/{len(gt_files)}] MISSING {gt.name}")
            continue

        eval1 = evaluate_resolutions(gt, pred, args.fuzzy)
        eval2 = analyze_turn_phrase(gt, pred)

        out = out_dir / f"{gt.stem}_eval.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "gt": str(gt),
                    "pred": str(pred),
                    "phrase_based": eval1,
                    "turn_phrase_based": eval2,
                },
                f,
                indent=2,
            )

        tpr.append(eval1["tp_rate"])
        fnr.append(eval1["fn_rate"])
        prec.append(eval2["precision"])
        rec.append(eval2["recall"])

        evaluated += 1
        print(f"[{i}/{len(gt_files)}] OK {gt.name}")

    print("\n=== SUMMARY ===")
    print(f"Evaluated: {evaluated}")
    print(f"Missing:   {missing}")

    if evaluated:
        mtpr, stpr = summarize(tpr)
        mfnr, sfnr = summarize(fnr)
        mp, sp = summarize(prec)
        mr, sr = summarize(rec)

        print(f"TPR: {mtpr*100:.2f}% ± {stpr*100:.2f}%")
        print(f"FNR: {mfnr*100:.2f}% ± {sfnr*100:.2f}%")
        print(f"Prec: {mp*100:.2f}% ± {sp*100:.2f}%")
        print(f"Recall: {mr*100:.2f}% ± {sr*100:.2f}%")

    return 0 if evaluated else 2


if __name__ == "__main__":
    raise SystemExit(main())
