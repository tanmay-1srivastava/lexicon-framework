#!/usr/bin/env python3
"""Evaluate V2 approach with comprehensive metrics"""

import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

DATA_DIR = Path('../../data_generation/new_data/generated_datasets')
V2_DIR = Path('resolution_results_v2')
COT_DIR = Path('../../baselines/cot_one_sided_results')

def exact_match(pred, gt):
    """Check if prediction exactly matches ground truth"""
    return pred.strip().lower() == gt.strip().lower()

def semantic_similarity(pred, gt):
    """Calculate semantic similarity using embeddings"""
    pred_emb = model.encode([pred])
    gt_emb = model.encode([gt])
    sim = cosine_similarity(pred_emb, gt_emb)[0][0]
    return sim * 100  # Convert to percentage

def calculate_bleu_like_score(predicted, ground_truth):
    """Calculate F1 token overlap"""
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
    return f1 * 100

def evaluate_approach(approach_dir, dataset_suffix, approach_name, common_files_only=None, common_turns_only=None):
    """Evaluate an approach against ground truth"""
    
    exact_matches = 0
    semantic_sims = []
    bleu_scores = []
    total_gt = 0
    total_predicted = 0
    
    files_to_eval = common_files_only if common_files_only else []
    
    for result_file in sorted(approach_dir.glob(f'*{dataset_suffix}')):
        # Filter to common files if specified
        if common_files_only:
            base_name = result_file.stem.replace(dataset_suffix.replace('.json', ''), '')
            if base_name not in common_files_only:
                continue
        
        with open(result_file) as f:
            results = json.load(f)
        
        # Load ground truth
        gt_file = DATA_DIR / results['file']
        with open(gt_file) as f:
            gt_data = json.load(f)
        
        # Create lookup for predictions
        predictions = {}
        for res in results.get('ground_truth_resolutions', []):
            turn_id = res.get('turn_id') or res.get('trigger_turn_id')
            predictions[turn_id] = res['resolved_entity']
        
        # Count predictions
        total_predicted += len(predictions)
        
        # Evaluate against ground truth
        for gt_res in gt_data['ground_truth_resolutions']:
            turn_id = gt_res['trigger_turn_id']
            gt_entity = gt_res['resolved_entity']
            
            # Filter to User A spatial/temporal only
            source = gt_res.get('resolution_source', '')
            if 'User A' not in source and 'Conversation Context' not in source:
                continue
            
            # CRITICAL: Only evaluate turns that this approach actually attempted
            if turn_id not in predictions:
                continue
            
            # If common_turns specified, only evaluate those
            if common_turns_only:
                file_base = results['file'].replace('.json', '')
                if (file_base, turn_id) not in common_turns_only:
                    continue
            
            total_gt += 1
            pred = predictions[turn_id]
            
            # Exact match
            if exact_match(pred, gt_entity):
                exact_matches += 1
            
            # Semantic similarity
            sem_sim = semantic_similarity(pred, gt_entity)
            semantic_sims.append(sem_sim)
            
            # BLEU
            bleu = calculate_bleu_like_score(pred, gt_entity)
            bleu_scores.append(bleu)
    
    # Calculate averages
    exact_rate = (exact_matches / total_gt * 100) if total_gt > 0 else 0
    avg_semantic = sum(semantic_sims) / len(semantic_sims) if semantic_sims else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    high_semantic = sum(1 for s in semantic_sims if s >= 85) / len(semantic_sims) * 100 if semantic_sims else 0
    
    print(f"\nOVERALL RESULTS FOR {approach_name}")
    print(f"Exact Match Rate:           {exact_rate:.1f}%")
    print(f"Semantic Similarity (avg):  {avg_semantic:.1f}%")
    print(f"High Semantic (≥85%):       {high_semantic:.1f}%")
    print(f"BLEU Score (avg):           {avg_bleu:.1f}%")
    print(f"Evaluated Resolutions:      {total_gt} (out of {total_predicted} predicted)")
    
    return {
        'name': approach_name,
        'exact_match': exact_rate,
        'semantic_sim': avg_semantic,
        'high_semantic': high_semantic,
        'bleu': avg_bleu,
        'gt_count': total_gt,
        'predicted_count': total_predicted
    }

def main():
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION: V2 vs CoT Baseline")
    print("=" * 70)
    
    # Find common files
    v2_files = {f.stem.replace('_resolution_v2', '') for f in V2_DIR.glob('*_resolution_v2.json')}
    cot_files = {f.stem.replace('_cot_one_sided_result', '') for f in COT_DIR.glob('*_cot_one_sided_result.json')}
    common_files = sorted(v2_files.intersection(cot_files))
    
    print(f"\n📊 FAIR COMPARISON ON {len(common_files)} COMMON FILES:")
    print(f"   {', '.join(common_files)}")
    
    # First, find common turns between CoT and V2
    print("\n🔍 Finding common turns between approaches...")
    common_turns = set()
    for file_base in common_files:
        cot_file = COT_DIR / f"{file_base}_cot_one_sided_result.json"
        v2_file = V2_DIR / f"{file_base}_resolution_v2.json"
        
        if cot_file.exists() and v2_file.exists():
            with open(cot_file) as f:
                cot_data = json.load(f)
            with open(v2_file) as f:
                v2_data = json.load(f)
            
            cot_turns = {r['trigger_turn_id'] for r in cot_data.get('ground_truth_resolutions', [])}
            v2_turns = {r['turn_id'] for r in v2_data.get('ground_truth_resolutions', [])}
            
            overlap = cot_turns.intersection(v2_turns)
            for turn in overlap:
                common_turns.add((file_base, turn))
    
    print(f"   Found {len(common_turns)} common turn resolutions\n")
    
    # Evaluate CoT baseline
    print("\n" + "=" * 70)
    print("EVALUATING: CoT One-Sided Baseline")
    print("=" * 70)
    cot_results = evaluate_approach(
        COT_DIR, 
        '_cot_one_sided_result.json',
        'CoT One-Sided Baseline',
        common_files,
        common_turns
    )
    
    # Evaluate V2
    print("\n" + "=" * 70)
    print("EVALUATING: Our Improved V2 Approach")
    print("=" * 70)
    v2_results = evaluate_approach(
        V2_DIR,
        '_resolution_v2.json', 
        'Our Improved V2 Approach',
        common_files,
        common_turns
    )
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (FAIR - SAME FILES & SAME TURNS)")
    print("=" * 70)
    print(f"{'Metric':<25} {'CoT Baseline':<15} {'Our V2':<15} {'Δ':<10}")
    print("-" * 70)
    print(f"{'Exact Match':<25} {cot_results['exact_match']:>6.1f}%        {v2_results['exact_match']:>6.1f}%        {v2_results['exact_match']-cot_results['exact_match']:+6.1f}%")
    print(f"{'Semantic Similarity':<25} {cot_results['semantic_sim']:>6.1f}%        {v2_results['semantic_sim']:>6.1f}%        {v2_results['semantic_sim']-cot_results['semantic_sim']:+6.1f}%")
    print(f"{'High Semantic (≥85%)':<25} {cot_results['high_semantic']:>6.1f}%        {v2_results['high_semantic']:>6.1f}%        {v2_results['high_semantic']-cot_results['high_semantic']:+6.1f}%")
    print(f"{'BLEU Score':<25} {cot_results['bleu']:>6.1f}%        {v2_results['bleu']:>6.1f}%        {v2_results['bleu']-cot_results['bleu']:+6.1f}%")
    print(f"{'Evaluated Resolutions':<25} {cot_results['gt_count']:>6}          {v2_results['gt_count']:>6}")
    print("=" * 70)
    
    # Determine winner
    v2_wins = sum([
        v2_results['exact_match'] > cot_results['exact_match'],
        v2_results['semantic_sim'] > cot_results['semantic_sim'],
        v2_results['bleu'] > cot_results['bleu']
    ])
    
    print(f"\n🏆 WINNER: ", end="")
    if v2_wins >= 2:
        print("Our Improved V2 Approach!")
    else:
        print("CoT Baseline")

if __name__ == '__main__':
    main()
