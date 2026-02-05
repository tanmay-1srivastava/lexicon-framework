#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, '.')
from improved_processor import ImprovedResolutionProcessor

# Load work_collaboration_002 (0% BLEU - worst case)
with open('../../data_generation/new_data/generated_datasets/work_collaboration_002.json') as f:
    data = json.load(f)

transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A'][:20]
context = data['mobile_context_snapshot']['user_a']

spatial_temporal_phrases = ['here', 'there', 'this', 'that', 'tomorrow', 'today', 'then', 
                            'later', 'soon', 'next', 'now', 'upstairs', 'downstairs',
                            'building', 'room', 'floor', 'office', 'location', 'place']

gt_all = [r for r in data['ground_truth_resolutions'] if 'User A' in r['resolution_source']]
gt_filtered = [r for r in gt_all if any(phrase in r.get('ambiguous_phrase', '').lower() for phrase in spatial_temporal_phrases)]

print('WORK_COLLABORATION_002 (0% BLEU - WHY?):')
print('='*80)
print()
print(f'Ground Truth (spatial/temporal): {len(gt_filtered)}')
for r in gt_filtered:
    print(f'  Turn {r["trigger_turn_id"]}: "{r["ambiguous_phrase"]}" → "{r["resolved_entity"]}"')
print()

proc = ImprovedResolutionProcessor()
preds = proc.resolve_references(transcript, context, '2024-11-17T09:30:00')

print(f'Predictions: {len(preds)}')
for p in preds:
    print(f'  Turn {p["turn_id"]}: "{p["ambiguous_phrase"]}" → "{p["resolved_entity"]}"')
print()

print('='*80)
print('ANALYSIS:')
print('='*80)
for gt in gt_filtered:
    gt_turn = gt['trigger_turn_id']
    gt_phrase = gt['ambiguous_phrase'].lower()
    match = [p for p in preds if p['turn_id'] == gt_turn and gt_phrase in p['ambiguous_phrase'].lower()]
    if match:
        print(f'Turn {gt_turn} "{gt_phrase}": MATCHED')
    else:
        print(f'Turn {gt_turn} "{gt_phrase}": MISSING')
        # Check why
        turn_preds = [p for p in preds if p['turn_id'] == gt_turn]
        if turn_preds:
            print(f'  Turn has other predictions: {[p["ambiguous_phrase"] for p in turn_preds]}')
        else:
            print(f'  Turn has NO predictions')
