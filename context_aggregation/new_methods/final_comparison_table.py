#!/usr/bin/env python3
"""Final comprehensive table: All baselines + our system"""

import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

DATA_DIR = Path('../../data_generation/new_data/generated_datasets')
BASELINES_DIR = Path('../../baselines')

# All approaches to evaluate
APPROACHES = [
    {
        'name': 'CoT Basic',
        'dir': BASELINES_DIR / 'cot_basic_results',
        'suffix': '_cot_result.json'
    },
    {
        'name': 'CoT Self-Reflexion',
        'dir': BASELINES_DIR / 'cot_self_reflexion',
        'suffix': '_cot_self_reflexion_result.json'
    },
    {
        'name': 'ToT Basic',
        'dir': BASELINES_DIR / 'tot_basic_results',
        'suffix': '_tot_result.json'
    },
    {
        'name': 'CoT One-Sided',
        'dir': BASELINES_DIR / 'cot_one_sided_results',
        'suffix': '_cot_one_sided_result.json'
    },
    {
        'name': 'Our V2',
        'dir': Path('resolution_results_v2'),
        'suffix': '_resolution_v2.json'
    }
]

def semantic_similarity(pred, gt):
    """Calculate semantic similarity using embeddings"""
    pred_emb = model.encode([pred])
    gt_emb = model.encode([gt])
    sim = cosine_similarity(pred_emb, gt_emb)[0][0]
    return sim * 100

def evaluate_approach(approach_dir, dataset_suffix):
    """Evaluate coverage and accuracy for an approach"""
    
    if not approach_dir.exists():
        return None
    
    total_gt_references = 0
    found_references = 0
    semantic_sims = []
    
    for result_file in sorted(approach_dir.glob(f'*{dataset_suffix}')):
        with open(result_file) as f:
            results = json.load(f)
        
        # Load ground truth
        gt_file = DATA_DIR / results['file']
        if not gt_file.exists():
            continue
            
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
            
            total_gt_references += 1
            
            if turn_id in predictions:
                found_references += 1
                pred = predictions[turn_id]
                
                # Semantic similarity
                sem_sim = semantic_similarity(pred, gt_entity)
                semantic_sims.append(sem_sim)
    
    # Calculate metrics
    coverage = (found_references / total_gt_references * 100) if total_gt_references > 0 else 0
    avg_semantic = sum(semantic_sims) / len(semantic_sims) if semantic_sims else 0
    
    return {
        'total_gt': total_gt_references,
        'found': found_references,
        'coverage': coverage,
        'semantic_sim': avg_semantic
    }

def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "FINAL COMPARISON TABLE")
    print("=" * 80)
    
    results = []
    
    for approach in APPROACHES:
        print(f"\nEvaluating {approach['name']}...", end=" ")
        res = evaluate_approach(approach['dir'], approach['suffix'])
        
        if res:
            results.append({
                'name': approach['name'],
                **res
            })
            print(f"✓ ({res['found']}/{res['total_gt']} references)")
        else:
            print("✗ (not found)")
    
    # Print table
    print("\n" + "=" * 80)
    print(f"{'Approach':<25} {'Coverage':<15} {'Semantic Sim':<15} {'Found/Total':<15}")
    print("=" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['coverage']:>6.1f}%        {r['semantic_sim']:>6.1f}%        {r['found']:>4}/{r['total_gt']:<4}")
    
    print("=" * 80)
    
    # Find best in each category
    best_coverage = max(results, key=lambda x: x['coverage'])
    best_semantic = max(results, key=lambda x: x['semantic_sim'])
    
    print("\n🏆 BEST PERFORMERS:")
    print(f"   Coverage:           {best_coverage['name']:<20} ({best_coverage['coverage']:.1f}%)")
    print(f"   Semantic Similarity: {best_semantic['name']:<20} ({best_semantic['semantic_sim']:.1f}%)")
    
    # Overall score (balanced)
    print("\n📊 BALANCED SCORE (Coverage × Semantic Sim):")
    for r in results:
        balanced = (r['coverage'] * r['semantic_sim']) / 100
        r['balanced'] = balanced
        print(f"   {r['name']:<25} {balanced:>6.1f}")
    
    best_balanced = max(results, key=lambda x: x['balanced'])
    print(f"\n🥇 OVERALL WINNER: {best_balanced['name']} ({best_balanced['balanced']:.1f})")
    
    print("=" * 80)
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    our_v2 = next((r for r in results if r['name'] == 'Our V2'), None)
    if our_v2:
        cot_basic = next((r for r in results if r['name'] == 'CoT Basic'), None)
        cot_one_sided = next((r for r in results if r['name'] == 'CoT One-Sided'), None)
        
        if cot_basic:
            print(f"   • Our V2 has {our_v2['coverage'] - cot_basic['coverage']:+.1f}% coverage vs CoT Basic")
        if cot_one_sided:
            print(f"   • Our V2 has {our_v2['coverage'] - cot_one_sided['coverage']:+.1f}% coverage vs CoT One-Sided")
            print(f"   • Our V2 finds {our_v2['found'] - cot_one_sided['found']} MORE references than CoT One-Sided")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()
