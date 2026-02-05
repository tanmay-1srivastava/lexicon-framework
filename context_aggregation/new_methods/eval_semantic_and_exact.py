#!/usr/bin/env python3
"""
Comprehensive evaluation using:
1. Exact string match (strict)
2. Semantic similarity (embedding-based)
3. BLEU score (token overlap)

Compares CoT one-sided baseline vs our improved approach
"""

import json
import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Spatial/temporal filter phrases
SPATIAL_TEMPORAL_PHRASES = [
    "here", "there", "this room", "that building", "upstairs", "downstairs",
    "this building", "that floor", "this office", "that location",
    "now", "then", "tomorrow", "later", "today", "tonight", "next week",
    "soon", "earlier", "after", "before", "at that time",
    "server room", "data center", "this folder", "that directory",
    "this cable", "that cable", "this map", "that kit",
    "up", "down", "station", "just down the hallway"
]

# Load semantic model
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


def is_spatial_temporal(phrase):
    """Check if phrase is spatial or temporal"""
    phrase_lower = phrase.lower()
    return any(p in phrase_lower for p in SPATIAL_TEMPORAL_PHRASES)


def exact_match(pred, gt):
    """Strict exact string match (case-insensitive, stripped)"""
    pred_clean = pred.lower().strip()
    gt_clean = gt.lower().strip()
    return pred_clean == gt_clean


def semantic_similarity(pred, gt):
    """Calculate semantic similarity using embeddings"""
    if not pred or not gt:
        return 0.0
    
    # Get embeddings
    pred_emb = model.encode([pred])
    gt_emb = model.encode([gt])
    
    # Cosine similarity
    similarity = cosine_similarity(pred_emb, gt_emb)[0][0]
    return float(similarity)


def calculate_bleu_like_score(predicted, ground_truth):
    """F1-based BLEU score"""
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return set(text.split())
    
    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)
    
    if not gt_tokens:
        return 0.0
    
    matches = pred_tokens.intersection(gt_tokens)
    precision = len(matches) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(matches) / len(gt_tokens) if gt_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_approach(results_dir, dataset_dir, approach_name, common_files_only=None):
    """Evaluate an approach using all three metrics
    
    Args:
        common_files_only: If provided, only evaluate these dataset names (without extension)
    """
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {approach_name}")
    print(f"{'='*80}\n")
    
    total_exact = 0
    total_semantic = 0.0
    total_bleu = 0.0
    total_gt_count = 0
    total_pred_count = 0
    
    high_semantic_matches = 0  # semantic >= 0.85
    
    dataset_results = []
    
    # Get all result files
    if 'cot_one_sided' in str(results_dir).lower():
        pattern = '_cot_one_sided_result.json'
    else:
        pattern = '_resolution.json'
    
    result_files = sorted([f for f in os.listdir(results_dir) 
                          if f.endswith(pattern) and not f.startswith('batch')])
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return None
    
    for result_file in result_files:
        result_path = os.path.join(results_dir, result_file)
        
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        # Get dataset name
        dataset_name = result_file.replace('_cot_one_sided_result.json', '').replace('_resolution.json', '')
        
        # Filter by common files if specified
        if common_files_only and dataset_name not in common_files_only:
            continue
        
        dataset_file = f"{dataset_name}.json"
        dataset_path = os.path.join(dataset_dir, dataset_file)
        
        if not os.path.exists(dataset_path):
            print(f"Skipping {dataset_name} - dataset not found")
            continue
        
        # Load ground truth FROM DATASET
        with open(dataset_path, 'r') as f:
            gt_data = json.load(f)
        
        # ACTUAL ground truth from dataset
        actual_gt = gt_data.get('ground_truth_resolutions', [])
        
        # PREDICTIONS from approach
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
        
        # Calculate metrics
        dataset_exact = 0
        dataset_semantic = 0.0
        dataset_bleu = 0.0
        dataset_high_semantic = 0
        
        gt_count = len(filtered_gt)
        pred_count = len(predictions)
        
        for gt in filtered_gt:
            gt_turn_id = gt.get('trigger_turn_id')
            gt_phrase = gt.get('ambiguous_phrase', '')
            gt_resolved = gt.get('resolved_entity', '')
            
            # Find matching prediction
            best_exact = False
            best_semantic = 0.0
            best_bleu = 0.0
            
            for pred in predictions:
                pred_turn_id = pred.get('trigger_turn_id') or pred.get('turn_id')
                pred_phrase = pred.get('ambiguous_phrase', '')
                pred_resolved = pred.get('resolved_entity', '')
                
                # Match by turn_id and phrase
                if pred_turn_id == gt_turn_id and pred_phrase.lower() == gt_phrase.lower():
                    # Calculate all metrics
                    is_exact = exact_match(pred_resolved, gt_resolved)
                    sem_score = semantic_similarity(pred_resolved, gt_resolved)
                    bleu_score = calculate_bleu_like_score(pred_resolved, gt_resolved)
                    
                    best_exact = best_exact or is_exact
                    best_semantic = max(best_semantic, sem_score)
                    best_bleu = max(best_bleu, bleu_score)
            
            dataset_exact += 1 if best_exact else 0
            dataset_semantic += best_semantic
            dataset_bleu += best_bleu
            
            if best_semantic >= 0.85:
                dataset_high_semantic += 1
        
        # Calculate averages
        exact_rate = (dataset_exact / gt_count * 100) if gt_count > 0 else 0.0
        avg_semantic = (dataset_semantic / gt_count * 100) if gt_count > 0 else 0.0
        avg_bleu = (dataset_bleu / gt_count * 100) if gt_count > 0 else 0.0
        high_sem_rate = (dataset_high_semantic / gt_count * 100) if gt_count > 0 else 0.0
        
        dataset_results.append({
            'dataset': dataset_name,
            'exact_match': exact_rate,
            'semantic_sim': avg_semantic,
            'bleu': avg_bleu,
            'high_semantic': high_sem_rate,
            'gt_count': gt_count,
            'pred_count': pred_count
        })
        
        total_exact += dataset_exact
        total_semantic += dataset_semantic
        total_bleu += dataset_bleu
        total_gt_count += gt_count
        total_pred_count += pred_count
        high_semantic_matches += dataset_high_semantic
        
        print(f"{dataset_name:30} Exact:{exact_rate:6.1f}%  Semantic:{avg_semantic:6.1f}%  BLEU:{avg_bleu:6.1f}%  (GT:{gt_count}, Pred:{pred_count})")
    
    # Overall statistics
    overall_exact = (total_exact / total_gt_count * 100) if total_gt_count > 0 else 0.0
    overall_semantic = (total_semantic / total_gt_count * 100) if total_gt_count > 0 else 0.0
    overall_bleu = (total_bleu / total_gt_count * 100) if total_gt_count > 0 else 0.0
    high_sem_overall = (high_semantic_matches / total_gt_count * 100) if total_gt_count > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS FOR {approach_name}")
    print(f"{'='*80}")
    print(f"Exact Match Rate:           {overall_exact:.1f}%")
    print(f"Semantic Similarity (avg):  {overall_semantic:.1f}%")
    print(f"High Semantic Match (≥85%): {high_sem_overall:.1f}%")
    print(f"BLEU Score (avg):           {overall_bleu:.1f}%")
    print(f"Total GT (spatial/temporal, User A only): {total_gt_count}")
    print(f"Total Predictions: {total_pred_count}")
    print(f"{'='*80}\n")
    
    return {
        'approach_name': approach_name,
        'exact_match': overall_exact,
        'semantic_similarity': overall_semantic,
        'high_semantic_match': high_sem_overall,
        'bleu': overall_bleu,
        'total_gt': total_gt_count,
        'total_pred': total_pred_count,
        'datasets': dataset_results
    }


def main():
    base_dir = Path(__file__).parent
    dataset_dir = base_dir.parent.parent / 'data_generation' / 'new_data' / 'generated_datasets'
    baselines_dir = base_dir.parent.parent / 'baselines'
    
    # Find common files between CoT baseline and our approach
    cot_dir = baselines_dir / 'cot_one_sided_results'
    our_dir = base_dir / 'resolution_results'
    
    cot_files = set([f.replace('_cot_one_sided_result.json', '') 
                     for f in os.listdir(cot_dir) if f.endswith('_cot_one_sided_result.json')])
    our_files = set([f.replace('_resolution.json', '') 
                     for f in os.listdir(our_dir) if f.endswith('_resolution.json')])
    
    common_files = sorted(cot_files.intersection(our_files))
    
    print(f"\n{'='*100}")
    print(f"FAIR COMPARISON: Evaluating on {len(common_files)} COMMON FILES ONLY")
    print(f"{'='*100}")
    print(f"Common files: {', '.join(common_files)}\n")
    
    approaches = [
        {
            'name': 'CoT One-Sided Baseline',
            'dir': cot_dir
        },
        {
            'name': 'Our Improved Approach',
            'dir': our_dir
        }
    ]
    
    all_results = []
    
    for approach in approaches:
        if not approach['dir'].exists():
            print(f"Skipping {approach['name']} - directory not found: {approach['dir']}")
            continue
        
        result = evaluate_approach(str(approach['dir']), str(dataset_dir), 
                                  approach['name'], common_files_only=common_files)
        if result:
            all_results.append(result)
    
    # Comparison summary
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Approach':<30} {'Exact Match':<15} {'Semantic Sim':<15} {'High Sem(≥85%)':<15} {'BLEU':<10} {'GT Count':<10}")
    print("-"*100)
    
    for result in all_results:
        print(f"{result['approach_name']:<30} {result['exact_match']:>6.1f}% {' '*8} "
              f"{result['semantic_similarity']:>6.1f}% {' '*8} {result['high_semantic_match']:>6.1f}% {' '*8} "
              f"{result['bleu']:>6.1f}% {' '*3} {result['total_gt']:<10}")
    
    print("="*100)
    print("\nMETRIC DEFINITIONS:")
    print("  Exact Match:      Strict string equality (case-insensitive)")
    print("  Semantic Sim:     Average embedding cosine similarity (0-100%)")
    print("  High Sem (≥85%):  Percentage with semantic similarity ≥ 0.85")
    print("  BLEU:             Token overlap F1 score")
    print("="*100)
    
    # Save results
    output_file = base_dir / 'comprehensive_eval_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': all_results,
            'timestamp': '2026-02-05T16:00:00',
            'note': 'Evaluation using exact match, semantic similarity, and BLEU'
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
