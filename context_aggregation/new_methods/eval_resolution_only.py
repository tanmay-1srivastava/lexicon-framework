#!/usr/bin/env python3
"""
Focused evaluation: ONLY context resolution (spatial/temporal)
Does NOT re-run info gap detection
"""

import json
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, '.')

from improved_processor import ImprovedResolutionProcessor

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

def evaluate_resolution_dataset(dataset_path):
    """Evaluate spatial/temporal resolution on one dataset"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    # ONE-SIDED: Only User A's messages
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A']
    context = data['mobile_context_snapshot']['user_a']
    
    # Use generation_metadata if available
    ref_time = data.get('generation_metadata', {}).get('timestamp', '2024-11-17T09:30:00')
    
    # FILTER: Only spatial/temporal ground truth (exclude person/object references)
    spatial_temporal_phrases = ['here', 'there', 'this', 'that', 'tomorrow', 'today', 'then', 
                                'later', 'soon', 'next', 'now', 'upstairs', 'downstairs',
                                'building', 'room', 'floor', 'office', 'location', 'place']
    
    # CRITICAL FIX: Only include GT where speaker actually matches User A
    gt_resolutions = []
    for r in data['ground_truth_resolutions']:
        # Check if marked as User A source
        if not r.get('resolution_source', '').startswith('User A'):
            continue
        
        # Check if spatial/temporal
        if not any(phrase in r.get('ambiguous_phrase', '').lower() 
                   for phrase in spatial_temporal_phrases):
            continue
        
        # VERIFY speaker actually is User A (bug fix for mislabeled data)
        turn_id = r.get('trigger_turn_id')
        turn = next((t for t in data['conversation_transcript'] if t['turn_id'] == turn_id), None)
        
        if turn and turn['speaker'] == 'User A':
            gt_resolutions.append(r)
        # else: skip mislabeled ground truth
    
    if not gt_resolutions:
        return None
    
    # Resolve references with IMPROVED processor
    processor = ImprovedResolutionProcessor()
    backstory = data.get('backstory', {})
    predicted = processor.resolve_references(transcript, context, ref_time, backstory)
    
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
    print('IMPROVED RESOLUTION - FOCUSED EVALUATION')
    print('='*80)
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Datasets: {len(DATASETS)}')
    print()
    print('IMPROVEMENTS:')
    print('  1. One-sided analysis (only User A messages, no other user info)')
    print('  2. Demonstrative objects with spatial context ("this map", "that kit")')
    print('  3. Better temporal resolution from calendar_next')
    print('  4. False positive prevention (skip if already clear)')
    print()
    
    # Evaluate resolution
    print('='*80)
    print('SPATIAL/TEMPORAL RESOLUTION:')
    print('='*80)
    print()
    
    resolution_results = []
    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / dataset_name
        print(f'Evaluating {dataset_name}...', end=' ', flush=True)
        
        try:
            result = evaluate_resolution_dataset(dataset_path)
            if result:
                resolution_results.append(result)
                print(f'✓ BLEU={result["avg_bleu_score"]*100:.1f}% (GT:{result["ground_truth"]}, Pred:{result["predicted"]})')
            else:
                print('⊘ No spatial/temporal references')
        except Exception as e:
            print(f'✗ Error: {e}')
            import traceback
            traceback.print_exc()
    
    print()
    
    if resolution_results:
        total_gt_res = sum(r['ground_truth'] for r in resolution_results)
        total_matches = sum(r['matches'] for r in resolution_results)
        avg_bleu = sum(r['avg_bleu_score'] * r['ground_truth'] for r in resolution_results) / total_gt_res if total_gt_res > 0 else 0
        
        print('='*80)
        print('RESOLUTION SUMMARY:')
        print('='*80)
        print('NOTE: Ground truth filtered to ONLY spatial/temporal references')
        print('      One-sided analysis: User A messages only, no other user info')
        print()
        print(f'Average BLEU Score: {avg_bleu*100:.1f}% (was 24.5% before improvements)')
        print(f'Match Rate (≥50% BLEU): {total_matches/total_gt_res*100:.1f}%' if total_gt_res > 0 else 'Match Rate: N/A')
        print(f'Total ground truth (spatial/temporal only): {total_gt_res}')
        print(f'Total predicted: {sum(r["predicted"] for r in resolution_results)}')
        print(f'Total matches: {total_matches}')
        print()
        
        total_spatial = sum(r['spatial_count'] for r in resolution_results)
        total_temporal = sum(r['temporal_count'] for r in resolution_results)
        print(f'Spatial resolutions: {total_spatial}')
        print(f'Temporal resolutions: {total_temporal}')
        print()
        
        # Compare to previous
        improvement = avg_bleu*100 - 24.5
        print('='*80)
        print('IMPROVEMENT:')
        print('='*80)
        print(f'  Previous BLEU: 24.5%')
        print(f'  Current BLEU: {avg_bleu*100:.1f}%')
        print(f'  Change: {improvement:+.1f}%')
        print()
        
        if avg_bleu >= 0.60:
            print('✅ TARGET ACHIEVED: BLEU ≥ 60%')
        elif avg_bleu >= 0.50:
            print('🎯 GOOD PROGRESS: BLEU ≥ 50%')
        else:
            print(f'⚠️  Need {60 - avg_bleu*100:.1f}% more for 60% target')
        print()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'spatial_temporal_resolution': {
            'average_bleu_score': avg_bleu if resolution_results else 0,
            'by_dataset': resolution_results
        }
    }
    
    output_path = Path(f'resolution_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'Results saved to: {output_path}')
    print()

if __name__ == "__main__":
    main()
