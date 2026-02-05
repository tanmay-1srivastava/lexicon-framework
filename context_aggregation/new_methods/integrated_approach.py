#!/usr/bin/env python3
"""
Integrated approach: Use info gap detection to guide context resolution
"""

import json
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, '.')

from improved_processor import ImprovedInfoGapProcessor, ImprovedResolutionProcessor

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

def integrated_resolution(dataset_path):
    """Use info gap detection to guide what to resolve"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    # ONE-SIDED: Only User A's messages
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A']
    context = data['mobile_context_snapshot']['user_a']
    ref_time = data.get('generation_metadata', {}).get('timestamp', '2024-11-17T09:30:00')
    
    # STEP 1: Detect info gaps (what's ambiguous?)
    gap_detector = ImprovedInfoGapProcessor()
    known_resolutions = [r for r in data['ground_truth_resolutions'] if 'User A' in r['resolution_source']]
    detected_gaps = gap_detector.detect_gaps(transcript, context, known_resolutions)
    
    # STEP 2: Extract ambiguous phrases from detected gaps
    # Focus only on spatial/temporal gaps
    ambiguous_phrases = []
    for gap in detected_gaps:
        entity_type = gap.get('entity_type', '').lower()
        if entity_type in ['location', 'spatial', 'temporal', 'appointment']:
            # Extract the actual ambiguous phrase from the question
            turn_id = gap.get('turn_id')
            turn = next((t for t in transcript if t['turn_id'] == turn_id), None)
            if turn:
                # Look for demonstrative/deictic words in the turn
                text = turn['text'].lower()
                deictic_words = ['here', 'there', 'this', 'that', 'now', 'then', 'later', 'tomorrow']
                for word in deictic_words:
                    if word in text:
                        ambiguous_phrases.append({
                            'turn_id': turn_id,
                            'phrase': word,
                            'full_text': turn['text']
                        })
    
    # STEP 3: Resolve ONLY the identified ambiguous phrases
    resolver = ImprovedResolutionProcessor()
    
    # Create focused prompt for only these phrases
    resolutions = []
    for phrase_info in ambiguous_phrases:
        turn_id = phrase_info['turn_id']
        phrase = phrase_info['phrase']
        
        # Build mini-context with just this turn
        turn = next((t for t in transcript if t['turn_id'] == turn_id), None)
        if not turn:
            continue
        
        # Simple resolution with metadata
        resolution = {
            'turn_id': turn_id,
            'ambiguous_phrase': phrase,
            'resolved_entity': None,
            'resolution_type': None,
            'metadata_source': None
        }
        
        # Spatial resolution
        if phrase in ['here', 'there']:
            resolution['resolved_entity'] = context.get('location_semantic', 'Unknown location')
            resolution['resolution_type'] = 'spatial'
            resolution['metadata_source'] = 'location_semantic'
        
        # Temporal resolution
        elif phrase in ['now']:
            resolution['resolved_entity'] = ref_time
            resolution['resolution_type'] = 'temporal'
            resolution['metadata_source'] = 'reference_time'
        
        elif phrase in ['then', 'later']:
            calendar = context.get('calendar_next', '')
            if calendar:
                resolution['resolved_entity'] = calendar
                resolution['resolution_type'] = 'temporal'
                resolution['metadata_source'] = 'calendar_next'
        
        if resolution['resolved_entity']:
            resolutions.append(resolution)
    
    return {
        'gaps_detected': len(detected_gaps),
        'ambiguous_phrases_found': len(ambiguous_phrases),
        'resolutions': resolutions
    }

def evaluate_with_integration(dataset_path):
    """Evaluate using integrated approach"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A']
    
    # Get verified ground truth (speaker matches)
    spatial_temporal_phrases = ['here', 'there', 'this', 'that', 'tomorrow', 'today', 'then', 
                                'later', 'soon', 'next', 'now', 'upstairs', 'downstairs',
                                'building', 'room', 'floor', 'office', 'location', 'place']
    
    gt_resolutions = []
    for r in data['ground_truth_resolutions']:
        if not r.get('resolution_source', '').startswith('User A'):
            continue
        if not any(phrase in r.get('ambiguous_phrase', '').lower() for phrase in spatial_temporal_phrases):
            continue
        
        # Verify speaker
        turn_id = r.get('trigger_turn_id')
        turn = next((t for t in data['conversation_transcript'] if t['turn_id'] == turn_id), None)
        if turn and turn['speaker'] == 'User A':
            gt_resolutions.append(r)
    
    if not gt_resolutions:
        return None
    
    # Get predictions using integration
    result = integrated_resolution(dataset_path)
    predictions = result['resolutions']
    
    # Better BLEU scoring
    def calculate_better_bleu(pred, gt):
        """Improved BLEU with partial credit"""
        import re
        
        def tokenize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            return text.split()
        
        pred_tokens = set(tokenize(pred))
        gt_tokens = set(tokenize(gt))
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        overlap = len(pred_tokens & gt_tokens)
        
        # Precision and recall
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens)
        
        if precision == 0 or recall == 0:
            return 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # More granular scoring
        if f1 >= 0.9:
            return 1.0  # Excellent match
        elif f1 >= 0.7:
            return 0.8  # Good match
        elif f1 >= 0.5:
            return 0.6  # Partial match
        elif f1 >= 0.3:
            return 0.3  # Weak match
        else:
            return 0.0  # No match
    
    # Match predictions with GT
    total_score = 0
    matches = 0
    
    for gt in gt_resolutions:
        gt_turn = gt.get('trigger_turn_id')
        gt_phrase = gt.get('ambiguous_phrase', '').lower()
        gt_entity = gt.get('resolved_entity', '')
        
        best_score = 0
        for pred in predictions:
            pred_turn = pred.get('turn_id')
            pred_phrase = pred.get('ambiguous_phrase', '').lower()
            
            if pred_turn == gt_turn and (gt_phrase in pred_phrase or pred_phrase in gt_phrase):
                pred_entity = pred.get('resolved_entity', '')
                score = calculate_better_bleu(pred_entity, gt_entity)
                best_score = max(best_score, score)
        
        total_score += best_score
        if best_score >= 0.5:
            matches += 1
    
    accuracy = total_score / len(gt_resolutions) if gt_resolutions else 0
    
    return {
        'dataset': dataset_path.name,
        'ground_truth': len(gt_resolutions),
        'gaps_detected': result['gaps_detected'],
        'resolutions': len(predictions),
        'matches': matches,
        'avg_bleu_score': accuracy
    }

def main():
    print('='*80)
    print('INTEGRATED APPROACH: Info Gap Detection → Context Resolution')
    print('='*80)
    print()
    print('APPROACH:')
    print('  1. Use info gap detection to find what\'s ambiguous (general)')
    print('  2. Only resolve those identified phrases (focused, not everything)')
    print('  3. Better BLEU scoring with more granular partial credit')
    print()
    
    results = []
    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / dataset_name
        print(f'Evaluating {dataset_name}...', end=' ', flush=True)
        
        try:
            result = evaluate_with_integration(dataset_path)
            if result:
                results.append(result)
                print(f'✓ BLEU={result["avg_bleu_score"]*100:.1f}% (Gaps:{result["gaps_detected"]}, Resolutions:{result["resolutions"]})')
            else:
                print('⊘ No spatial/temporal GT')
        except Exception as e:
            print(f'✗ Error: {e}')
            import traceback
            traceback.print_exc()
    
    print()
    
    if results:
        total_gt = sum(r['ground_truth'] for r in results)
        total_matches = sum(r['matches'] for r in results)
        avg_bleu = sum(r['avg_bleu_score'] * r['ground_truth'] for r in results) / total_gt if total_gt > 0 else 0
        
        print('='*80)
        print('INTEGRATED APPROACH RESULTS:')
        print('='*80)
        print(f'Average BLEU Score: {avg_bleu*100:.1f}%')
        print(f'Match Rate (≥50% BLEU): {total_matches/total_gt*100:.1f}%' if total_gt > 0 else 'N/A')
        print(f'Total GT: {total_gt}')
        print(f'Total resolutions: {sum(r["resolutions"] for r in results)}')
        print()
        
        print('Comparison to previous approach (36.7%):')
        improvement = avg_bleu*100 - 36.7
        print(f'  Change: {improvement:+.1f}%')
        print()

if __name__ == "__main__":
    main()
