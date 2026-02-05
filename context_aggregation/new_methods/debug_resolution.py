#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, '.')
from improved_processor import ImprovedResolutionProcessor

# Load test case
with open('../../data_generation/new_data/generated_datasets/doctor_visit_001.json') as f:
    data = json.load(f)

transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A'][:20]
context = data['mobile_context_snapshot']['user_a']
ref_time = '2024-11-17T09:30:00'

# Get ground truth
gt = [r for r in data['ground_truth_resolutions'] if 'User A' in r['resolution_source']][:10]

print('GROUND TRUTH (first 10):')
for g in gt:
    print(f"  Turn {g['trigger_turn_id']}: '{g['ambiguous_phrase']}' → '{g['resolved_entity']}'")

# Run processor
proc = ImprovedResolutionProcessor()
preds = proc.resolve_references(transcript, context, ref_time)

print()
print(f'PREDICTIONS ({len(preds)} total):')
for p in preds[:10]:
    print(f"  Turn {p['turn_id']}: '{p['ambiguous_phrase']}' → '{p['resolved_entity']}'")

print()
print('MATCHING:')
for g in gt[:5]:
    gt_turn = g['trigger_turn_id']
    gt_phrase = g['ambiguous_phrase'].lower()
    gt_entity = g['resolved_entity']
    
    # Find matches
    matches = [p for p in preds if p['turn_id'] == gt_turn and gt_phrase in p['ambiguous_phrase'].lower()]
    
    if matches:
        for m in matches:
            score = proc.calculate_bleu_like_score(m['resolved_entity'], gt_entity)
            print(f"Turn {gt_turn} '{gt_phrase}':")
            print(f"  GT: {gt_entity}")
            print(f"  Pred: {m['resolved_entity']}")
            print(f"  BLEU: {score:.2f}")
    else:
        print(f"Turn {gt_turn} '{gt_phrase}': NO MATCH")
