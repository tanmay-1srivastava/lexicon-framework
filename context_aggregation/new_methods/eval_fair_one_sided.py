#!/usr/bin/env python3
"""Fair evaluation: Only count references resolvable from User A's perspective"""

import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

DATA_DIR = Path('../../data_generation/new_data/generated_datasets')
V2_DIR = Path('resolution_results_v2')
COT_DIR = Path('../../baselines/cot_one_sided_results')

def semantic_similarity(pred, gt):
    pred_emb = model.encode([pred])
    gt_emb = model.encode([gt])
    sim = cosine_similarity(pred_emb, gt_emb)[0][0]
    return sim * 100

def is_resolvable_one_sided(gt_ref, user_a_turns):
    """Check if reference can be resolved with ONLY User A's messages"""
    
    source = gt_ref.get('resolution_source', '')
    
    # GPS/Calendar/Wifi - resolvable with metadata
    if any(x in source for x in ['GPS', 'Calendar', 'Wifi']):
        return True
    
    # "Conversation Context" - need to check if it refers to User A's own messages
    # or if it requires User B's messages
    if 'Conversation Context' in source:
        turn_id = gt_ref['trigger_turn_id']
        phrase = gt_ref['ambiguous_phrase'].lower()
        
        # Get User A's turns before this one
        previous_a_turns = [t for t in user_a_turns if t['turn_id'] < turn_id]
        
        # If phrase is "there", "that", "it" etc. - check if User A mentioned it
        # in their own previous messages. If not, it likely refers to User B.
        if phrase in ['there', 'that', 'it', 'them', 'those', 'this']:
            # Very likely refers to something User B just said
            # Skip these - can't resolve without User B's messages
            return False
        
        # If it's a time reference like "now", "then", "later" - usually resolvable
        if phrase in ['now', 'then', 'later', 'tomorrow', 'today', 'yesterday']:
            return True
        
        # Default: assume resolvable
        return True
    
    return True

def evaluate_approach_fair(approach_dir, dataset_suffix, approach_name, common_files_only=None):
    """Evaluate only on references resolvable from one-sided perspective"""
    
    total_resolvable = 0
    found_references = 0
    semantic_sims = []
    
    for result_file in sorted(approach_dir.glob(f'*{dataset_suffix}')):
        if common_files_only:
            base_name = result_file.stem.replace(dataset_suffix.replace('.json', ''), '')
            if base_name not in common_files_only:
                continue
        
        with open(result_file) as f:
            results = json.load(f)
        
        gt_file = DATA_DIR / results['file']
        if not gt_file.exists():
            continue
            
        with open(gt_file) as f:
            gt_data = json.load(f)
        
        # Get User A turns
        user_a_turns = [t for t in gt_data['conversation_transcript'] if t['speaker'] == 'User A']
        
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
            
            # ONLY User A references
            if 'User A' not in source and 'Conversation Context' not in source:
                continue
            
            # CRITICAL: Skip references that require User B's messages
            if not is_resolvable_one_sided(gt_res, user_a_turns):
                continue
            
            total_resolvable += 1
            
            if turn_id in predictions:
                found_references += 1
                pred = predictions[turn_id]
                sem_sim = semantic_similarity(pred, gt_entity)
                semantic_sims.append(sem_sim)
    
    coverage = (found_references / total_resolvable * 100) if total_resolvable > 0 else 0
    avg_semantic = sum(semantic_sims) / len(semantic_sims) if semantic_sims else 0
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {approach_name}")
    print(f"{'='*70}")
    print(f"\n📊 COVERAGE (on resolvable references only):")
    print(f"   Resolvable References:   {total_resolvable}")
    print(f"   Found & Attempted:       {found_references}")
    print(f"   Coverage Rate:           {coverage:.1f}%")
    print(f"   Missed:                  {total_resolvable - found_references}")
    
    print(f"\n🎯 ACCURACY:")
    print(f"   Semantic Similarity:     {avg_semantic:.1f}%")
    
    return {
        'name': approach_name,
        'total_resolvable': total_resolvable,
        'found': found_references,
        'coverage': coverage,
        'semantic_sim': avg_semantic
    }

def main():
    print("=" * 70)
    print("FAIR ONE-SIDED EVALUATION")
    print("(Only counting references resolvable without User B's messages)")
    print("=" * 70)
    
    # Find common files
    v2_files = {f.stem.replace('_resolution_v2', '') for f in V2_DIR.glob('*_resolution_v2.json')}
    cot_files = {f.stem.replace('_cot_one_sided_result', '') for f in COT_DIR.glob('*_cot_one_sided_result.json')}
    common_files = sorted(v2_files.intersection(cot_files))
    
    print(f"\n📁 Evaluating on {len(common_files)} common files:")
    print(f"   {', '.join(common_files)}")
    
    # Evaluate CoT baseline
    cot_results = evaluate_approach_fair(
        COT_DIR, 
        '_cot_one_sided_result.json',
        'CoT One-Sided Baseline',
        common_files
    )
    
    # Evaluate V2
    v2_results = evaluate_approach_fair(
        V2_DIR,
        '_resolution_v2.json', 
        'Our Improved V2 Approach',
        common_files
    )
    
    # Comparison
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON (FAIR)")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'CoT':<15} {'Our V2':<15} {'Winner':<10}")
    print("-" * 70)
    
    cov_winner = "V2 ✓" if v2_results['coverage'] > cot_results['coverage'] else "CoT ✓"
    print(f"{'Coverage Rate':<30} {cot_results['coverage']:>6.1f}%        {v2_results['coverage']:>6.1f}%        {cov_winner}")
    
    sem_winner = "V2 ✓" if v2_results['semantic_sim'] > cot_results['semantic_sim'] else "CoT ✓"
    print(f"{'Semantic Similarity':<30} {cot_results['semantic_sim']:>6.1f}%        {v2_results['semantic_sim']:>6.1f}%        {sem_winner}")
    
    print("-" * 70)
    print(f"{'References found':<30} {cot_results['found']:>6}          {v2_results['found']:>6}")
    print(f"{'Resolvable references':<30} {cot_results['total_resolvable']:>6}          {v2_results['total_resolvable']:>6}")
    
    print("\n" + "=" * 70)
    
    # Target achievement
    print("\n🎯 TARGET ACHIEVEMENT:")
    print(f"   Coverage target: 75%  → V2 at {v2_results['coverage']:.1f}% ({'✓' if v2_results['coverage'] >= 75 else '✗'})")
    print(f"   Semantic target: 65%  → V2 at {v2_results['semantic_sim']:.1f}% ({'✓' if v2_results['semantic_sim'] >= 65 else '✗'})")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
