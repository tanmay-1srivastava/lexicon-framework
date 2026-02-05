#!/usr/bin/env python3
"""Evaluate both COVERAGE (did we find all references?) and ACCURACY (were we correct?)"""

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
    return sim * 100

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

def evaluate_approach_full(approach_dir, dataset_suffix, approach_name, common_files_only=None):
    """Evaluate both coverage and accuracy"""
    
    # COVERAGE METRICS
    total_gt_references = 0  # Total User A spatial/temporal in ground truth
    found_references = 0      # How many did we attempt to resolve?
    
    # ACCURACY METRICS (for found references only)
    exact_matches = 0
    semantic_sims = []
    bleu_scores = []
    
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
        
        # Evaluate against ground truth
        for gt_res in gt_data['ground_truth_resolutions']:
            turn_id = gt_res['trigger_turn_id']
            gt_entity = gt_res['resolved_entity']
            source = gt_res.get('resolution_source', '')
            
            # ONLY User A spatial/temporal references
            if 'User A' not in source and 'Conversation Context' not in source:
                continue
            
            # This is a valid ground truth reference we should find
            total_gt_references += 1
            
            # Did we attempt to resolve it?
            if turn_id in predictions:
                found_references += 1
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
    
    # Calculate metrics
    coverage = (found_references / total_gt_references * 100) if total_gt_references > 0 else 0
    exact_rate = (exact_matches / found_references * 100) if found_references > 0 else 0
    avg_semantic = sum(semantic_sims) / len(semantic_sims) if semantic_sims else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {approach_name}")
    print(f"{'='*70}")
    print(f"\n📊 COVERAGE (Did we find all references?):")
    print(f"   Ground Truth References: {total_gt_references}")
    print(f"   Found & Attempted:       {found_references}")
    print(f"   Coverage Rate:           {coverage:.1f}%")
    print(f"   Missed:                  {total_gt_references - found_references}")
    
    print(f"\n🎯 ACCURACY (For the {found_references} we attempted):")
    print(f"   Exact Match:             {exact_rate:.1f}%")
    print(f"   Semantic Similarity:     {avg_semantic:.1f}%")
    print(f"   BLEU Score:              {avg_bleu:.1f}%")
    
    return {
        'name': approach_name,
        'total_gt': total_gt_references,
        'found': found_references,
        'coverage': coverage,
        'exact_match': exact_rate,
        'semantic_sim': avg_semantic,
        'bleu': avg_bleu
    }

def main():
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION: COVERAGE + ACCURACY")
    print("=" * 70)
    
    # Find common files
    v2_files = {f.stem.replace('_resolution_v2', '') for f in V2_DIR.glob('*_resolution_v2.json')}
    cot_files = {f.stem.replace('_cot_one_sided_result', '') for f in COT_DIR.glob('*_cot_one_sided_result.json')}
    common_files = sorted(v2_files.intersection(cot_files))
    
    print(f"\n📁 Evaluating on {len(common_files)} common files:")
    print(f"   {', '.join(common_files)}")
    
    # Evaluate CoT baseline
    cot_results = evaluate_approach_full(
        COT_DIR, 
        '_cot_one_sided_result.json',
        'CoT One-Sided Baseline',
        common_files
    )
    
    # Evaluate V2
    v2_results = evaluate_approach_full(
        V2_DIR,
        '_resolution_v2.json', 
        'Our Improved V2 Approach',
        common_files
    )
    
    # Comparison
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'CoT':<15} {'Our V2':<15} {'Winner':<10}")
    print("-" * 70)
    
    # Coverage
    cov_winner = "V2 ✓" if v2_results['coverage'] > cot_results['coverage'] else "CoT ✓"
    print(f"{'Coverage Rate':<30} {cot_results['coverage']:>6.1f}%        {v2_results['coverage']:>6.1f}%        {cov_winner}")
    
    # Accuracy metrics
    exact_winner = "V2 ✓" if v2_results['exact_match'] > cot_results['exact_match'] else "CoT ✓"
    print(f"{'Exact Match (accuracy)':<30} {cot_results['exact_match']:>6.1f}%        {v2_results['exact_match']:>6.1f}%        {exact_winner}")
    
    sem_winner = "V2 ✓" if v2_results['semantic_sim'] > cot_results['semantic_sim'] else "CoT ✓"
    print(f"{'Semantic Similarity':<30} {cot_results['semantic_sim']:>6.1f}%        {v2_results['semantic_sim']:>6.1f}%        {sem_winner}")
    
    bleu_winner = "V2 ✓" if v2_results['bleu'] > cot_results['bleu'] else "CoT ✓"
    print(f"{'BLEU Score':<30} {cot_results['bleu']:>6.1f}%        {v2_results['bleu']:>6.1f}%        {bleu_winner}")
    
    print("-" * 70)
    print(f"{'References found':<30} {cot_results['found']:>6}          {v2_results['found']:>6}")
    print(f"{'Total GT references':<30} {cot_results['total_gt']:>6}          {v2_results['total_gt']:>6}")
    
    # Overall winner
    print("\n" + "=" * 70)
    v2_score = sum([
        v2_results['coverage'] > cot_results['coverage'],
        v2_results['exact_match'] > cot_results['exact_match'],
        v2_results['semantic_sim'] > cot_results['semantic_sim'],
        v2_results['bleu'] > cot_results['bleu']
    ])
    
    print(f"🏆 OVERALL: ", end="")
    if v2_score >= 3:
        print("Our V2 Wins! (better on ≥3 metrics)")
    elif v2_score >= 2:
        print("TIE (each wins 2 metrics)")
    else:
        print("CoT Wins (better on ≥3 metrics)")
    
    # Key insight
    print("\n💡 KEY INSIGHT:")
    if v2_results['coverage'] > cot_results['coverage'] and v2_results['exact_match'] < cot_results['exact_match']:
        print("   V2 finds MORE references but CoT is MORE ACCURATE on what it finds")
    elif v2_results['coverage'] < cot_results['coverage'] and v2_results['exact_match'] > cot_results['exact_match']:
        print("   CoT finds MORE references but V2 is MORE ACCURATE on what it finds")
    elif v2_results['coverage'] > cot_results['coverage'] and v2_results['exact_match'] > cot_results['exact_match']:
        print("   V2 is BETTER on BOTH coverage AND accuracy!")
    else:
        print("   CoT is BETTER on BOTH coverage AND accuracy")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
