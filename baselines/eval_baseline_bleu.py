#!/usr/bin/env python3
"""
Evaluate BLEU scores for baseline approaches (CoT Basic, CoT Self-Reflexion, ToT)
Compares baseline context resolution with improved approach
"""

import json
import os
import re
from pathlib import Path

# Spatial/temporal filter phrases (same as improved approach)
SPATIAL_TEMPORAL_PHRASES = [
    "here", "there", "this room", "that building", "upstairs", "downstairs",
    "this building", "that floor", "this office", "that location",
    "now", "then", "tomorrow", "later", "today", "tonight", "next week",
    "soon", "earlier", "after", "before", "at that time",
    "server room", "data center", "this folder", "that directory",
    "this cable", "that cable", "this map", "that kit",
    "up", "down", "station", "just down the hallway"
]

def calculate_bleu_like_score(predicted, ground_truth):
    """Calculate F1-based BLEU score with 5-level granularity (same as improved approach)"""
    
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return set(text.split())
    
    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)
    
    if not gt_tokens:
        return 0.0
    
    # F1 calculation
    matches = pred_tokens.intersection(gt_tokens)
    precision = len(matches) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(matches) / len(gt_tokens) if gt_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Granular scoring
    if f1 >= 0.9:
        return 1.0  # Excellent (90%+)
    elif f1 >= 0.7:
        return 0.9  # Very good (70-90%)
    elif f1 >= 0.5:
        return 0.7  # Good (50-70%)
    elif f1 >= 0.3:
        return 0.5  # Acceptable (30-50%)
    else:
        return 0.0  # Poor (<30%)


def is_spatial_temporal(phrase):
    """Check if phrase is spatial or temporal"""
    phrase_lower = phrase.lower()
    return any(p in phrase_lower for p in SPATIAL_TEMPORAL_PHRASES)


def evaluate_baseline_results(baseline_dir, dataset_dir, baseline_name):
    """Evaluate a baseline approach using BLEU scores"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {baseline_name}")
    print(f"{'='*80}\n")
    
    total_bleu = 0.0
    total_matches = 0
    total_gt_count = 0
    total_pred_count = 0
    dataset_results = []
    
    # Get all result files
    result_files = sorted([f for f in os.listdir(baseline_dir) if f.endswith('result.json') and not f.startswith('batch') and not f.startswith('pair')])
    
    for result_file in result_files:
        result_path = os.path.join(baseline_dir, result_file)
        
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        dataset_name = result_file.replace('_cot_result.json', '').replace('_tot_result.json', '').replace('_cot_self_reflexionresult.json', '').replace('_cot_one_sided_result.json', '')
        dataset_file = f"{dataset_name}.json"
        dataset_path = os.path.join(dataset_dir, dataset_file)
        
        # Load ground truth FROM DATASET
        with open(dataset_path, 'r') as f:
            gt_data = json.load(f)
        
        # ACTUAL ground truth from dataset
        actual_gt = gt_data.get('ground_truth_resolutions', [])
        
        # PREDICTIONS from baseline (stored as "ground_truth_resolutions" in baseline output)
        predictions = result_data.get('ground_truth_resolutions', [])
        
        # Filter GT to spatial/temporal + speaker verification
        transcript = gt_data.get('conversation_transcript', [])
        filtered_gt = []
        
        for gt in actual_gt:
            turn_id = gt.get('trigger_turn_id')
            ambiguous_phrase = gt.get('ambiguous_phrase', '')
            
            # Check if spatial/temporal
            if not is_spatial_temporal(ambiguous_phrase):
                continue
            
            # Speaker verification - only include User A's turns
            turn = next((t for t in transcript if t['turn_id'] == turn_id), None)
            if not turn or turn.get('speaker') != 'User A':
                continue
            
            filtered_gt.append(gt)
        
        # Calculate BLEU scores
        dataset_bleu = 0.0
        matches = 0
        gt_count = len(filtered_gt)
        pred_count = len(predictions)
        
        for gt in filtered_gt:
            gt_turn_id = gt.get('trigger_turn_id')
            gt_phrase = gt.get('ambiguous_phrase', '')
            gt_resolved = gt.get('resolved_entity', '')
            
            # Find matching prediction
            best_score = 0.0
            for pred in predictions:
                pred_turn_id = pred.get('trigger_turn_id') or pred.get('turn_id')
                pred_phrase = pred.get('ambiguous_phrase', '')
                pred_resolved = pred.get('resolved_entity', '')
                
                # Match by turn_id and phrase
                if pred_turn_id == gt_turn_id and pred_phrase.lower() == gt_phrase.lower():
                    score = calculate_bleu_like_score(pred_resolved, gt_resolved)
                    best_score = max(best_score, score)
            
            dataset_bleu += best_score
            if best_score >= 0.7:  # Good match threshold
                matches += 1
        
        # Calculate average BLEU for this dataset
        avg_bleu = (dataset_bleu / gt_count * 100) if gt_count > 0 else 0.0
        match_rate = (matches / gt_count * 100) if gt_count > 0 else 0.0
        
        dataset_results.append({
            'dataset': dataset_name,
            'bleu': avg_bleu,
            'matches': matches,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'match_rate': match_rate
        })
        
        total_bleu += dataset_bleu
        total_matches += matches
        total_gt_count += gt_count
        total_pred_count += pred_count
        
        print(f"{dataset_name:30} BLEU: {avg_bleu:6.1f}%  Matches: {matches}/{gt_count}  Predictions: {pred_count}")
    
    # Overall statistics
    overall_bleu = (total_bleu / total_gt_count * 100) if total_gt_count > 0 else 0.0
    overall_match_rate = (total_matches / total_gt_count * 100) if total_gt_count > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS FOR {baseline_name}")
    print(f"{'='*80}")
    print(f"Average BLEU Score: {overall_bleu:.1f}%")
    print(f"Match Rate (≥50% BLEU): {overall_match_rate:.1f}%")
    print(f"Total Ground Truth (spatial/temporal, User A only): {total_gt_count}")
    print(f"Total Predictions: {total_pred_count}")
    print(f"Total Matches: {total_matches}")
    print(f"{'='*80}\n")
    
    return {
        'baseline_name': baseline_name,
        'overall_bleu': overall_bleu,
        'match_rate': overall_match_rate,
        'total_gt': total_gt_count,
        'total_pred': total_pred_count,
        'total_matches': total_matches,
        'dataset_results': dataset_results
    }


def main():
    base_dir = Path(__file__).parent
    dataset_dir = base_dir.parent / 'data_generation' / 'new_data' / 'generated_datasets'
    
    baselines = [
        {
            'name': 'CoT Basic',
            'dir': base_dir / 'cot_basic_results'
        },
        {
            'name': 'CoT Self-Reflexion',
            'dir': base_dir / 'cot_self_reflexion'
        },
        {
            'name': 'ToT (Tree of Thoughts)',
            'dir': base_dir / 'tot_basic_results'
        },
        {
            'name': 'CoT One-Sided (User A Only)',
            'dir': base_dir / 'cot_one_sided_results'
        }
    ]
    
    all_results = []
    
    for baseline in baselines:
        if not baseline['dir'].exists():
            print(f"Skipping {baseline['name']} - directory not found: {baseline['dir']}")
            continue
        
        result = evaluate_baseline_results(str(baseline['dir']), str(dataset_dir), baseline['name'])
        all_results.append(result)
    
    # Comparison summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Baseline':<30} {'BLEU Score':<15} {'Match Rate':<15} {'GT Count':<12}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['baseline_name']:<30} {result['overall_bleu']:>6.1f}% {' '*8} "
              f"{result['match_rate']:>6.1f}% {' '*8} {result['total_gt']:<12}")
    
    print("-"*80)
    print(f"{'Improved Approach (Reference)':<30} {'40.4%':<15} {'46.7%':<15} {'45':<12}")
    print("="*80)
    
    # Save results
    output_file = base_dir / 'baseline_bleu_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'baseline_results': all_results,
            'improved_reference': {
                'bleu': 40.4,
                'match_rate': 46.7,
                'total_gt': 45
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
