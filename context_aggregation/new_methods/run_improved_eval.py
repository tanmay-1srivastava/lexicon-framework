#!/usr/bin/env python3
"""
Full evaluation script for improved processor - runs on all 9 datasets
Compares old vs improved performance
"""

import json
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, '.')

from improved_processor import ImprovedInfoGapProcessor, ImprovedResolutionProcessor
from openai import AzureOpenAI
import secret_keys

# Setup LLM for semantic matching
client = AzureOpenAI(
    api_version="2023-12-01-preview",
    api_key=secret_keys.Open_ai_key,
    azure_endpoint="https://initial-resources.cognitiveservices.azure.com/"
)

# Dataset paths
DATA_DIR = Path('../../data_generation/new_data/generated_datasets')
DATASETS = [
    'doctor_visit_001.json',
    'doctor_visit_002.json', 
    'doctor_visit_003.json',
    'friends_meeting_001.json',
    'friends_meeting_002.json',
    'friends_meeting_003.json',
    'work_collaboration_001.json',
    'work_collaboration_002.json',
    'work_collaboration_003.json'
]

def semantic_match(pred_gap, gt_gap):
    """LLM-based semantic matching for accurate evaluation (same as old evaluator)"""
    pred_q = pred_gap.get('question', '').strip()
    gt_q = gt_gap.get('natural_language_fallback', '').strip()
    
    if not pred_q or not gt_q:
        return False
    
    prompt = f"""Compare these two information gaps. Do they target the same missing data point?
Q1 (Predicted): {pred_q}
Q2 (Ground Truth): {gt_q}

Criteria:
- If Q1 is a confirmation of the entity in Q2, it is a MATCH.
- If they target the same Turn ID and the same noun/action, it is a MATCH.

Return JSON: {{"match": true/false, "reason": "..."}}"""
    
    try:
        res = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(res.choices[0].message.content)
        return result.get('match', False)
    except Exception as e:
        # Fallback to simple token matching
        pred_tokens = set(pred_q.lower().split())
        gt_tokens = set(gt_q.lower().split())
        overlap = len(pred_tokens & gt_tokens)
        if overlap == 0:
            return False
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1 >= 0.4

def evaluate_dataset(dataset_path):
    """Evaluate one dataset using HIGH_VALUE protocol queries as ground truth"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    # Get User A data (only User A has consistent ground truth)
    target = "User A"
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == target]
    context = data['mobile_context_snapshot']['user_a']
    
    # Get HIGH_VALUE protocol queries for User A
    hp_gt = [q for q in data['required_protocol_queries'] 
             if q['query_quality_check'] == 'HIGH_VALUE' and 
             any(t['turn_id'] == q['trigger_turn_id'] and t['speaker'] == target for t in transcript)]
    
    if not hp_gt:
        return None
    
    # Get known ground truth resolutions for context
    known_gt = [g for g in data['ground_truth_resolutions'] if target in g['resolution_source']]
    
    # Detect gaps with improved processor
    processor = ImprovedInfoGapProcessor()
    detected_gaps = processor.detect_gaps(transcript, context, known_gt)
    
    # Match with ground truth using semantic matching
    matched_gts = set()
    false_positives = 0
    
    for pred_gap in detected_gaps:
        matched = False
        for i, gt_gap in enumerate(hp_gt):
            if i not in matched_gts and semantic_match(pred_gap, gt_gap):
                matched_gts.add(i)
                matched = True
                break
        
        if not matched:
            false_positives += 1
    
    true_positives = len(matched_gts)
    false_negatives = len(hp_gt) - true_positives
    
    # Calculate metrics
    tpr = true_positives / len(hp_gt) if hp_gt else 0
    fnr = false_negatives / len(hp_gt) if hp_gt else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    return {
        'dataset': dataset_path.name,
        'ground_truth': len(hp_gt),
        'detected': len(detected_gaps),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'tpr': tpr,
        'fnr': fnr,
        'precision': precision,
        'priority_breakdown': {
            'critical': len([g for g in detected_gaps if g.get('priority') == 'CRITICAL']),
            'high': len([g for g in detected_gaps if g.get('priority') == 'HIGH']),
            'medium': len([g for g in detected_gaps if g.get('priority') == 'MEDIUM'])
        }
    }

def evaluate_resolution_dataset(dataset_path):
    """Evaluate spatial/temporal resolution on one dataset"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A']
    context = data['mobile_context_snapshot']['user_a']
    
    # Use generation_metadata if available, otherwise default
    ref_time = data.get('generation_metadata', {}).get('timestamp', '2024-11-17T09:30:00')
    
    # FILTER: Only spatial/temporal ground truth (exclude person/object references)
    spatial_temporal_phrases = ['here', 'there', 'this', 'that', 'tomorrow', 'today', 'then', 
                                'later', 'soon', 'next', 'now', 'upstairs', 'downstairs',
                                'building', 'room', 'floor', 'office', 'location', 'place']
    
    gt_resolutions = [r for r in data['ground_truth_resolutions'] 
                     if r.get('resolution_source', '').startswith('User A') and
                     any(phrase in r.get('ambiguous_phrase', '').lower() 
                         for phrase in spatial_temporal_phrases)]
    
    if not gt_resolutions:
        return None
    
    # Resolve references
    processor = ImprovedResolutionProcessor()
    predicted = processor.resolve_references(transcript, context, ref_time)
    
    # Match predictions with ground truth using BLEU scoring
    total_score = 0
    matches = 0
    
    for gt in gt_resolutions:
        gt_turn = gt.get('trigger_turn_id')
        gt_phrase = gt.get('ambiguous_phrase', '').lower()
        gt_entity = gt.get('resolved_entity', '')
        
        # Find best matching prediction
        best_score = 0
        for pred in predicted:
            pred_turn = pred.get('turn_id')
            pred_phrase = pred.get('ambiguous_phrase', '').lower()
            
            # Check if turn and phrases match
            if pred_turn == gt_turn and (gt_phrase in pred_phrase or pred_phrase in gt_phrase):
                pred_entity = pred.get('resolved_entity', '')
                score = processor.calculate_bleu_like_score(pred_entity, gt_entity)
                best_score = max(best_score, score)
        
        total_score += best_score
        if best_score >= 0.5:  # Partial match counts
            matches += 1
    
    accuracy = total_score / len(gt_resolutions) if gt_resolutions else 0
    
    return {
        'dataset': dataset_path.name,
        'ground_truth': len(gt_resolutions),
        'predicted': len(predicted),
        'matches': matches,
        'avg_bleu_score': accuracy,
        'spatial_count': len([r for r in predicted if r.get('resolution_type') == 'spatial']),
        'temporal_count': len([r for r in predicted if r.get('resolution_type') == 'temporal'])
    }

def main():
    print('='*80)
    print('IMPROVED PROCESSOR - FULL EVALUATION')
    print('='*80)
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Datasets: {len(DATASETS)}')
    print()
    
    # Evaluate info gap detection
    print('='*80)
    print('PART 1: INFORMATION GAP DETECTION')
    print('='*80)
    print()
    
    results = []
    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / dataset_name
        print(f'Evaluating {dataset_name}...', end=' ', flush=True)
        
        try:
            result = evaluate_dataset(dataset_path)
            if result:
                results.append(result)
                print(f'✓ TPR={result["tpr"]*100:.1f}% Precision={result["precision"]*100:.1f}%')
            else:
                print('⊘ No HIGH_VALUE queries for User A')
        except Exception as e:
            print(f'✗ Error: {e}')
            import traceback
            traceback.print_exc()
    
    print()
    
    # Calculate aggregate metrics
    total_gt = sum(r['ground_truth'] for r in results)
    total_tp = sum(r['true_positives'] for r in results)
    total_fp = sum(r['false_positives'] for r in results)
    total_fn = sum(r['false_negatives'] for r in results)
    
    avg_tpr = total_tp / total_gt if total_gt > 0 else 0
    avg_fnr = total_fn / total_gt if total_gt > 0 else 0
    avg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    print('='*80)
    print('INFO GAP DETECTION SUMMARY:')
    print('='*80)
    print(f'Overall TPR: {avg_tpr*100:.1f}% (was 65.7% with old method)')
    print(f'Overall FNR: {avg_fnr*100:.1f}%')
    print(f'Overall Precision: {avg_precision*100:.1f}% (was 25% with old method)')
    print(f'Total ground truth gaps: {total_gt}')
    print(f'Total detected: {sum(r["detected"] for r in results)}')
    print(f'True positives: {total_tp}')
    print(f'False positives: {total_fp}')
    print(f'False negatives: {total_fn}')
    print()
    
    # Show by category
    categories = {
        'doctor_visit': [r for r in results if 'doctor_visit' in r['dataset']],
        'friends_meeting': [r for r in results if 'friends_meeting' in r['dataset']],
        'work_collaboration': [r for r in results if 'work_collaboration' in r['dataset']]
    }
    
    print('By Category:')
    for cat_name, cat_results in categories.items():
        cat_gt = sum(r['ground_truth'] for r in cat_results)
        cat_tp = sum(r['true_positives'] for r in cat_results)
        cat_tpr = cat_tp / cat_gt if cat_gt > 0 else 0
        print(f'  {cat_name}: TPR={cat_tpr*100:.1f}%')
    print()
    
    # Evaluate spatial/temporal resolution
    print('='*80)
    print('PART 2: SPATIAL/TEMPORAL RESOLUTION')
    print('='*80)
    print()
    
    resolution_results = []
    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / dataset_name
        print(f'Evaluating {dataset_name}...', end=' ')
        
        try:
            result = evaluate_resolution_dataset(dataset_path)
            if result:
                resolution_results.append(result)
                print(f'✓ BLEU={result["avg_bleu_score"]*100:.1f}%')
            else:
                print('⊘ No spatial/temporal references')
        except Exception as e:
            print(f'✗ Error: {e}')
    
    print()
    
    if resolution_results:
        total_gt_res = sum(r['ground_truth'] for r in resolution_results)
        total_matches = sum(r['matches'] for r in resolution_results)
        avg_bleu = sum(r['avg_bleu_score'] * r['ground_truth'] for r in resolution_results) / total_gt_res if total_gt_res > 0 else 0
        
        print('='*80)
        print('SPATIAL/TEMPORAL RESOLUTION SUMMARY (FILTERED GT):')
        print('='*80)
        print('NOTE: Ground truth filtered to ONLY spatial/temporal references')
        print('      (excluded person/object references like "it", "her", "him")')
        print()
        print(f'Average BLEU Score: {avg_bleu*100:.1f}% (was ~30% with old method)')
        print(f'Match Rate (≥50% BLEU): {total_matches/total_gt_res*100:.1f}%' if total_gt_res > 0 else 'Match Rate: N/A')
        print(f'Total ground truth (spatial/temporal only): {total_gt_res}')
        print(f'Total predicted: {sum(r["predicted"] for r in resolution_results)}')
        print(f'Total matches: {total_matches}')
        print()
        
        total_spatial = sum(r['spatial_count'] for r in resolution_results)
        total_temporal = sum(r['temporal_count'] for r in resolution_results)
        total_temporal = sum(r['temporal_count'] for r in resolution_results)
        print(f'Spatial resolutions: {total_spatial}')
        print(f'Temporal resolutions: {total_temporal}')
        print()
    
    # Final comparison
    print('='*80)
    print('🎯 FINAL RESULTS:')
    print('='*80)
    print()
    print('OLD METHOD → IMPROVED METHOD:')
    print(f'  TPR: 65.7% → {avg_tpr*100:.1f}% ({avg_tpr*100 - 65.7:+.1f}%)')
    print(f'  Precision: 25% → {avg_precision*100:.1f}% ({avg_precision*100 - 25:+.1f}%)')
    if resolution_results:
        print(f'  Resolution BLEU (spatial/temporal only): ~30% → {avg_bleu*100:.1f}% ({avg_bleu*100 - 30:+.1f}%)')
    print()
    print('NOTE: Resolution evaluated on spatial/temporal refs only (fair comparison)')
    print()
    
    if avg_tpr >= 0.80:
        print('✅ TARGET ACHIEVED: TPR ≥ 80%')
    else:
        print(f'⚠️  Close to target: {80 - avg_tpr*100:.1f}% more needed')
    print()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'info_gap_detection': {
            'overall_tpr': avg_tpr,
            'overall_fnr': avg_fnr,
            'overall_precision': avg_precision,
            'total_ground_truth': total_gt,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'by_dataset': results
        },
        'spatial_temporal_resolution': {
            'average_bleu_score': avg_bleu if resolution_results else 0,
            'by_dataset': resolution_results
        }
    }
    
    output_path = Path(f'improved_eval_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'Results saved to: {output_path}')
    print()

if __name__ == "__main__":
    main()
