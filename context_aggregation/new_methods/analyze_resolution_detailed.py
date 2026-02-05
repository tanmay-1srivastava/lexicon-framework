#!/usr/bin/env python3
"""
Detailed resolution analysis with examples
"""

import json
import sys
sys.path.insert(0, '.')
from improved_processor import ImprovedResolutionProcessor

# Load a test case
with open('../../data_generation/new_data/generated_datasets/friends_meeting_002.json') as f:
    data = json.load(f)

transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A']
context = data['mobile_context_snapshot']['user_a']
ref_time = '2024-11-17T09:30:00'

# Filter GT to spatial/temporal only
spatial_temporal_phrases = ['here', 'there', 'this', 'that', 'tomorrow', 'today', 'then', 
                            'later', 'soon', 'next', 'now', 'upstairs', 'downstairs',
                            'building', 'room', 'floor', 'office', 'location', 'place']

gt_all = [r for r in data['ground_truth_resolutions'] if 'User A' in r['resolution_source']]
gt_filtered = [r for r in gt_all if any(phrase in r.get('ambiguous_phrase', '').lower() 
                                          for phrase in spatial_temporal_phrases)]

print('='*100)
print(f'RESOLUTION ANALYSIS: friends_meeting_002.json (50% BLEU - BEST PERFORMER)')
print('='*100)
print()

print(f'Total User A ground truth: {len(gt_all)}')
print(f'Spatial/temporal GT: {len(gt_filtered)}')
print(f'Excluded (person/object): {len(gt_all) - len(gt_filtered)}')
print()

print('EXCLUDED REFERENCES (person/object):')
excluded = [r for r in gt_all if r not in gt_filtered]
for r in excluded[:5]:
    print(f'  Turn {r["trigger_turn_id"]}: "{r["ambiguous_phrase"]}" → "{r["resolved_entity"]}"')
print()

print('SPATIAL/TEMPORAL GROUND TRUTH:')
for r in gt_filtered:
    print(f'  Turn {r["trigger_turn_id"]}: "{r["ambiguous_phrase"]}" → "{r["resolved_entity"]}"')
print()

# Run processor
proc = ImprovedResolutionProcessor()
preds = proc.resolve_references(transcript, context, ref_time)

print(f'PREDICTIONS: {len(preds)} total')
for p in preds:
    print(f'  Turn {p["turn_id"]}: "{p["ambiguous_phrase"]}" → "{p["resolved_entity"]}"')
    print(f'    Type: {p.get("resolution_type")}, Source: {p.get("metadata_source")}')
print()

print('='*100)
print('MATCHING ANALYSIS:')
print('='*100)

matches = 0
total_score = 0

for gt in gt_filtered:
    gt_turn = gt['trigger_turn_id']
    gt_phrase = gt['ambiguous_phrase'].lower()
    gt_entity = gt['resolved_entity']
    
    # Find matches
    pred_matches = [p for p in preds if p['turn_id'] == gt_turn and 
                    gt_phrase in p['ambiguous_phrase'].lower()]
    
    if pred_matches:
        for m in pred_matches:
            score = proc.calculate_bleu_like_score(m['resolved_entity'], gt_entity)
            total_score += score
            if score >= 0.5:
                matches += 1
            status = '✓ MATCH' if score >= 0.5 else '✗ PARTIAL'
            print(f'{status} Turn {gt_turn} "{gt_phrase}":')
            print(f'  GT:   "{gt_entity}"')
            print(f'  Pred: "{m["resolved_entity"]}"')
            print(f'  BLEU: {score:.2f}')
            print()
    else:
        print(f'✗ MISSING Turn {gt_turn} "{gt_phrase}":')
        print(f'  GT: "{gt_entity}"')
        print(f'  No prediction found')
        print()

avg_bleu = total_score / len(gt_filtered) if gt_filtered else 0
match_rate = matches / len(gt_filtered) if gt_filtered else 0

print('='*100)
print('SUMMARY:')
print('='*100)
print(f'Average BLEU: {avg_bleu*100:.1f}%')
print(f'Match Rate (≥50% BLEU): {match_rate*100:.1f}%')
print(f'Total GT: {len(gt_filtered)}')
print(f'Total Predictions: {len(preds)}')
print(f'Matched: {matches}')
print()
