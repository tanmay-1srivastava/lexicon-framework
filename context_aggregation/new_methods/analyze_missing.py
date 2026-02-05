#!/usr/bin/env python3
import json
from pathlib import Path

# Compare what CoT One-Sided found vs what we found
cot_file = Path('../../baselines/cot_one_sided_results/doctor_visit_002_cot_one_sided_result.json')
v2_file = Path('resolution_results_v2/doctor_visit_002_resolution_v2.json')
gt_file = Path('../../data_generation/new_data/generated_datasets/doctor_visit_002.json')

with open(cot_file) as f:
    cot = json.load(f)
with open(v2_file) as f:
    v2 = json.load(f)
with open(gt_file) as f:
    gt_data = json.load(f)

# Get User A references from GT
user_a_refs = [r for r in gt_data['ground_truth_resolutions'] 
               if 'User A' in r.get('resolution_source', '') or 
                  'Conversation Context' in r.get('resolution_source', '')]

cot_turns = {r['trigger_turn_id'] for r in cot['ground_truth_resolutions']}
v2_turns = {r['turn_id'] for r in v2['ground_truth_resolutions']}
gt_turns = {r['trigger_turn_id'] for r in user_a_refs}

print(f'Total User A references in GT: {len(gt_turns)}')
print(f'CoT found: {len(cot_turns)} ({len(cot_turns)/len(gt_turns)*100:.1f}%)')
print(f'V2 found: {len(v2_turns)} ({len(v2_turns)/len(gt_turns)*100:.1f}%)')

missing_from_v2 = gt_turns - v2_turns

print(f'\nMissing from V2: {len(missing_from_v2)} turns')
print('\nPhrases we missed:')
for r in user_a_refs:
    if r['trigger_turn_id'] in missing_from_v2:
        entity = r['resolved_entity']
        if len(entity) > 60:
            entity = entity[:60] + '...'
        print(f"  Turn {r['trigger_turn_id']}: '{r['ambiguous_phrase']}' → {entity}")
