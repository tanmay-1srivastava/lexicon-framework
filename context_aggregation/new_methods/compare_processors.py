#!/usr/bin/env python3
"""
Side-by-side comparison: Old vs Improved processors
"""

import json
import sys
sys.path.insert(0, '.')

from info_gap_processor import InfoGapProcessor
from improved_processor import ImprovedInfoGapProcessor

def compare_on_dataset(dataset_path):
    """Run both processors and compare"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    # Get User A data
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A'][:15]
    context = data['mobile_context_snapshot']['user_a']
    known_gt = [g for g in data['ground_truth_resolutions'] if 'User A' in g['resolution_source']]
    
    print('='*100)
    print(f'DATASET: {dataset_path}')
    print('='*100)
    print(f'Transcript: {len(transcript)} User A turns')
    print(f'Known resolutions: {len(known_gt)}')
    print()
    
    # OLD PROCESSOR
    print('-'*100)
    print('OLD PROCESSOR (whole conversation at once):')
    print('-'*100)
    old_proc = InfoGapProcessor()
    old_gaps = old_proc.detect_gaps(transcript, context, known_gt)
    
    print(f'Gaps detected: {len(old_gaps)}')
    print(f'Turns covered: {sorted(set([g.get("turn_id") for g in old_gaps if g.get("turn_id")]))}')
    print()
    print('Sample gaps:')
    for gap in old_gaps[:5]:
        print(f'  Turn {gap.get("turn_id")}: {gap.get("question", "N/A")[:80]}...')
    print()
    
    # IMPROVED PROCESSOR
    print('-'*100)
    print('IMPROVED PROCESSOR (turn-by-turn + priority filtering):')
    print('-'*100)
    improved_proc = ImprovedInfoGapProcessor()
    improved_gaps = improved_proc.detect_gaps(transcript, context, known_gt)
    
    print(f'Gaps detected: {len(improved_gaps)}')
    print(f'Turns covered: {sorted(set([g.get("turn_id") for g in improved_gaps if g.get("turn_id")]))}')
    print()
    print('Priority breakdown:')
    for priority in ['CRITICAL', 'HIGH', 'MEDIUM']:
        count = len([g for g in improved_gaps if g.get('priority') == priority])
        print(f'  {priority}: {count}')
    print()
    print('Sample gaps (top 5 CRITICAL/HIGH):')
    priority_gaps = [g for g in improved_gaps if g.get('priority') in ['CRITICAL', 'HIGH']]
    for gap in priority_gaps[:5]:
        pri = gap.get('priority')
        turn = gap.get('turn_id')
        entity = gap.get('entity_type', 'unknown')
        q = gap.get('question', 'N/A')[:70]
        print(f'  [{pri}] Turn {turn} ({entity}): {q}...')
    print()
    
    # COMPARISON
    print('='*100)
    print('COMPARISON:')
    print('='*100)
    old_turns = set([g.get("turn_id") for g in old_gaps if g.get("turn_id")])
    improved_turns = set([g.get("turn_id") for g in improved_gaps if g.get("turn_id")])
    
    print(f'Turn coverage: {len(old_turns)} → {len(improved_turns)} turns')
    print(f'Gaps detected: {len(old_gaps)} → {len(improved_gaps)}')
    
    new_turns = improved_turns - old_turns
    if new_turns:
        print(f'NEW turns covered by improved: {sorted(new_turns)}')
    
    print()

if __name__ == "__main__":
    datasets = [
        '../../data_generation/new_data/generated_datasets/doctor_visit_001.json',
        '../../data_generation/new_data/generated_datasets/work_collaboration_002.json',
        '../../data_generation/new_data/generated_datasets/friends_meeting_001.json'
    ]
    
    print()
    print('🔬 SIDE-BY-SIDE COMPARISON: OLD vs IMPROVED')
    print()
    
    for dataset in datasets:
        try:
            compare_on_dataset(dataset)
            print()
            print()
        except Exception as e:
            print(f'Error on {dataset}: {e}')
            import traceback
            traceback.print_exc()
