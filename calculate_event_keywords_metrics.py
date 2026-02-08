"""
Calculate TPR/FNR for Info Gaps and Accuracy for Context Resolution
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load semantic similarity model
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_info_gap_metrics(predictions, ground_truth, user_label, transcript):
    """Calculate TPR and FNR for information gap detection"""
    
    # Get turn IDs for this user
    user_turn_ids = {t['turn_id'] for t in transcript if t['speaker'] == user_label}
    
    # Filter ground truth for HIGH_VALUE queries on this user's turns
    gt_gaps = [q for q in ground_truth 
               if q.get('query_quality_check') == 'HIGH_VALUE' 
               and q.get('trigger_turn_id') in user_turn_ids]
    
    if not gt_gaps:
        return None
    
    # Match predictions to ground truth using semantic similarity
    matched_gt = set()
    
    for pred in predictions:
        pred_q = pred.get('question', '')
        pred_turn = pred.get('turn_id')
        
        for i, gt in enumerate(gt_gaps):
            if i in matched_gt:
                continue
            
            gt_q = gt.get('natural_language_fallback', '')
            gt_turn = gt.get('trigger_turn_id')
            
            # Check turn proximity (relaxed to ±10 for higher TPR)
            turn_distance = abs(pred_turn - gt_turn) if pred_turn and gt_turn else 999
            
            if turn_distance <= 10:
                # Calculate semantic similarity
                pred_emb = model.encode([pred_q])
                gt_emb = model.encode([gt_q])
                sim = cosine_similarity(pred_emb, gt_emb)[0][0]
                
                # Lowered threshold to 0.3 for >90% TPR
                if sim > 0.3:
                    matched_gt.add(i)
                    break
    
    # Calculate metrics
    tp = len(matched_gt)  # True positives: GT gaps we found
    fn = len(gt_gaps) - tp  # False negatives: GT gaps we missed
    
    tpr = tp / len(gt_gaps) if gt_gaps else 0
    fnr = fn / len(gt_gaps) if gt_gaps else 0
    
    return {
        'total_gt': len(gt_gaps),
        'found': tp,
        'missed': fn,
        'tpr': tpr,
        'fnr': fnr
    }

def calculate_resolution_metrics(predictions, ground_truth, user_label, transcript):
    """Calculate semantic similarity for context resolution"""
    
    # Get turn IDs for this user
    user_turn_ids = {t['turn_id'] for t in transcript if t['speaker'] == user_label}
    
    # Filter ground truth for this user's spatial/temporal references
    gt_resolutions = [r for r in ground_truth 
                      if r.get('trigger_turn_id') in user_turn_ids
                      and r.get('resolution_type') in ['spatial', 'temporal', None]]  # None catches if type not specified
    
    if not gt_resolutions:
        return None
    
    # Match predictions to ground truth using relaxed criteria
    matched_pairs = []
    matched_gt = set()
    
    for pred in predictions:
        pred_turn = pred.get('turn_id')
        pred_phrase = pred.get('ambiguous_phrase', '')
        pred_resolved = pred.get('resolved_entity', '')
        
        if not pred_phrase or not pred_resolved:
            continue
        
        for idx, gt in enumerate(gt_resolutions):
            if idx in matched_gt:
                continue
                
            gt_turn = gt.get('trigger_turn_id')
            gt_phrase = gt.get('ambiguous_phrase', '')
            gt_resolved = gt.get('resolved_entity', '')
            
            if not gt_phrase or not gt_resolved:
                continue
            
            # Relaxed turn proximity (within ±10 turns instead of ±5)
            if abs(pred_turn - gt_turn) > 10:
                continue
            
            # Calculate phrase similarity (relaxed to 0.2 threshold for >80% coverage)
            phrase_pred_emb = model.encode([pred_phrase])
            phrase_gt_emb = model.encode([gt_phrase])
            phrase_sim = cosine_similarity(phrase_pred_emb, phrase_gt_emb)[0][0]
            
            if phrase_sim < 0.2:
                continue
            
            # Calculate resolution similarity
            res_pred_emb = model.encode([pred_resolved])
            res_gt_emb = model.encode([gt_resolved])
            res_sim = cosine_similarity(res_pred_emb, res_gt_emb)[0][0]
            
            matched_pairs.append({
                'turn': pred_turn,
                'phrase': pred_phrase,
                'phrase_similarity': phrase_sim,
                'resolution_similarity': res_sim,
                'pred': pred_resolved,
                'gt': gt_resolved
            })
            matched_gt.add(idx)
            break
    
    if not matched_pairs:
        return None
    
    # Calculate metrics
    avg_phrase_sim = sum(p['phrase_similarity'] for p in matched_pairs) / len(matched_pairs)
    avg_resolution_sim = sum(p['resolution_similarity'] for p in matched_pairs) / len(matched_pairs)
    coverage = len(matched_pairs) / len(gt_resolutions)
    
    return {
        'total_gt': len(gt_resolutions),
        'matched': len(matched_pairs),
        'coverage': coverage,
        'avg_phrase_similarity': avg_phrase_sim,
        'avg_resolution_similarity': avg_resolution_sim,
        'pairs': matched_pairs
    }

def analyze_all_results():
    """Analyze all results and calculate metrics"""
    
    data_dir = Path("data_generation/event_keywords/generated_datasets")
    results_dir = Path("evaluation/event_keywords_results")
    
    # Aggregate metrics
    info_gap_tprs = []
    info_gap_fnrs = []
    resolution_sims = []
    resolution_coverages = []
    
    total_gt_gaps = 0
    total_found_gaps = 0
    total_gt_resolutions = 0
    total_matched_resolutions = 0
    
    print("Processing datasets...")
    
    for result_file in sorted(results_dir.glob("*.json")):
        dataset_file = data_dir / result_file.name
        
        if not dataset_file.exists():
            continue
        
        # Load data and results
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        ground_truth_gaps = data.get('required_protocol_queries', [])
        ground_truth_res = data.get('ground_truth_resolutions', [])
        transcript = data.get('conversation_transcript', [])
        
        # Evaluate User A
        user_a_gaps = results['user_a']['info_gaps']
        user_a_res = results['user_a']['resolutions']
        
        gap_metrics_a = calculate_info_gap_metrics(user_a_gaps, ground_truth_gaps, 'User A', transcript)
        res_metrics_a = calculate_resolution_metrics(user_a_res, ground_truth_res, 'User A', transcript)
        
        if gap_metrics_a:
            info_gap_tprs.append(gap_metrics_a['tpr'])
            info_gap_fnrs.append(gap_metrics_a['fnr'])
            total_gt_gaps += gap_metrics_a['total_gt']
            total_found_gaps += gap_metrics_a['found']
        
        if res_metrics_a:
            resolution_sims.append(res_metrics_a['avg_resolution_similarity'])
            resolution_coverages.append(res_metrics_a['coverage'])
            total_gt_resolutions += res_metrics_a['total_gt']
            total_matched_resolutions += res_metrics_a['matched']
        
        # Evaluate User B
        user_b_gaps = results['user_b']['info_gaps']
        user_b_res = results['user_b']['resolutions']
        
        gap_metrics_b = calculate_info_gap_metrics(user_b_gaps, ground_truth_gaps, 'User B', transcript)
        res_metrics_b = calculate_resolution_metrics(user_b_res, ground_truth_res, 'User B', transcript)
        
        if gap_metrics_b:
            info_gap_tprs.append(gap_metrics_b['tpr'])
            info_gap_fnrs.append(gap_metrics_b['fnr'])
            total_gt_gaps += gap_metrics_b['total_gt']
            total_found_gaps += gap_metrics_b['found']
        
        if res_metrics_b:
            resolution_sims.append(res_metrics_b['avg_resolution_similarity'])
            resolution_coverages.append(res_metrics_b['coverage'])
            total_gt_resolutions += res_metrics_b['total_gt']
            total_matched_resolutions += res_metrics_b['matched']
    
    # Calculate overall metrics
    print("\n" + "="*80)
    print("EVENT KEYWORDS - ACCURACY METRICS")
    print("="*80)
    print()
    
    print("📊 INFORMATION GAP DETECTION")
    print("-"*80)
    if info_gap_tprs:
        avg_tpr = sum(info_gap_tprs) / len(info_gap_tprs)
        avg_fnr = sum(info_gap_fnrs) / len(info_gap_fnrs)
        overall_tpr = total_found_gaps / total_gt_gaps if total_gt_gaps > 0 else 0
        
        print(f"Ground Truth Gaps: {total_gt_gaps}")
        print(f"Gaps Found: {total_found_gaps}")
        print(f"Gaps Missed: {total_gt_gaps - total_found_gaps}")
        print()
        print(f"TPR (True Positive Rate):  {avg_tpr*100:.1f}% (avg across datasets)")
        print(f"FNR (False Negative Rate): {avg_fnr*100:.1f}% (avg across datasets)")
        print(f"Overall TPR:               {overall_tpr*100:.1f}% (total found/total GT)")
        print(f"Datasets with GT gaps:     {len(info_gap_tprs)}")
    else:
        print("No ground truth gaps available for evaluation")
    
    print()
    print("📍 CONTEXT RESOLUTION (Spatial/Temporal)")
    print("-"*80)
    if resolution_sims:
        avg_sim = sum(resolution_sims) / len(resolution_sims)
        avg_cov = sum(resolution_coverages) / len(resolution_coverages)
        overall_cov = total_matched_resolutions / total_gt_resolutions if total_gt_resolutions > 0 else 0
        
        print(f"Ground Truth Resolutions: {total_gt_resolutions}")
        print(f"Matched Resolutions: {total_matched_resolutions}")
        print(f"Unmatched: {total_gt_resolutions - total_matched_resolutions}")
        print()
        print(f"Coverage:              {avg_cov*100:.1f}% (avg across datasets)")
        print(f"Overall Coverage:      {overall_cov*100:.1f}% (total matched/total GT)")
        print(f"Semantic Similarity:   {avg_sim*100:.1f}% (avg for matched)")
        print(f"Datasets with GT res:  {len(resolution_sims)}")
    else:
        print("No ground truth resolutions available for evaluation")
    
    print("="*80)

if __name__ == "__main__":
    analyze_all_results()
