#!/usr/bin/env python3
import json

# Load files
with open('resolution_results/doctor_visit_002_resolution_v2.json') as f:
    v2 = json.load(f)
with open('../../data_generation/new_data/generated_datasets/doctor_visit_002.json') as f:
    gt_data = json.load(f)
with open('../../baselines/cot_one_sided_results/doctor_visit_002_cot_one_sided_result.json') as f:
    cot = json.load(f)

# Get ground truth
gt_resolutions = {r['trigger_turn_id']: r['resolved_entity'] for r in gt_data['ground_truth_resolutions']}

print('TURN 5 COMPARISON:')
print(f'Ground Truth: {gt_resolutions.get(5)}')
cot_5 = next((r for r in cot['ground_truth_resolutions'] if r['trigger_turn_id'] == 5), None)
print(f'CoT Baseline: {cot_5["resolved_entity"] if cot_5 else "NOT FOUND"}')
v2_5 = next((r for r in v2['ground_truth_resolutions'] if r['turn_id'] == 5), None)
print(f'Our V2:       {v2_5["resolved_entity"] if v2_5 else "NOT FOUND"}')

print('\nTURN 17 COMPARISON:')
print(f'Ground Truth: {gt_resolutions.get(17)}')
cot_17 = next((r for r in cot['ground_truth_resolutions'] if r['trigger_turn_id'] == 17), None)
print(f'CoT Baseline: {cot_17["resolved_entity"] if cot_17 else "NOT FOUND"}')
v2_17 = next((r for r in v2['ground_truth_resolutions'] if r['turn_id'] == 17), None)
print(f'Our V2:       {v2_17["resolved_entity"] if v2_17 else "NOT FOUND"}')

print('\nTURN 27 COMPARISON:')
print(f'Ground Truth: {gt_resolutions.get(27)}')
cot_27 = next((r for r in cot['ground_truth_resolutions'] if r['trigger_turn_id'] == 27), None)
print(f'CoT Baseline: {cot_27["resolved_entity"] if cot_27 else "NOT FOUND"}')
v2_27 = next((r for r in v2['ground_truth_resolutions'] if r['turn_id'] == 27), None)
print(f'Our V2:       {v2_27["resolved_entity"] if v2_27 else "NOT FOUND"}')

print('\nTURN 51 COMPARISON:')
print(f'Ground Truth: {gt_resolutions.get(51)}')
cot_51 = next((r for r in cot['ground_truth_resolutions'] if r['trigger_turn_id'] == 51), None)
print(f'CoT Baseline: {cot_51["resolved_entity"] if cot_51 else "NOT FOUND"}')
v2_51 = next((r for r in v2['ground_truth_resolutions'] if r['turn_id'] == 51), None)
print(f'Our V2:       {v2_51["resolved_entity"] if v2_51 else "NOT FOUND"}')
