#!/usr/bin/env python3
import json

data = json.load(open('../../data_generation/new_data/generated_datasets/doctor_visit_001.json'))

# Check ground truth resolutions
all_gt = data['ground_truth_resolutions']
print(f'Total ground truth resolutions: {len(all_gt)}')
print()

# Group by user
user_a_gt = [r for r in all_gt if 'User A' in r['resolution_source']]
user_b_gt = [r for r in all_gt if 'User B' in r['resolution_source']]

print(f'User A resolutions: {len(user_a_gt)}')
print(f'User B resolutions: {len(user_b_gt)}')
print()

# Check which turns they're on
print('='*80)
print('CRITICAL CHECK: Do resolution sources match actual speakers?')
print('='*80)
print()

print('User A resolutions (first 10):')
mismatches = 0
for r in user_a_gt[:10]:
    turn_id = r['trigger_turn_id']
    turn = next((t for t in data['conversation_transcript'] if t['turn_id'] == turn_id), None)
    speaker = turn['speaker'] if turn else 'UNKNOWN'
    match = '✓' if speaker == 'User A' else '✗ MISMATCH'
    if speaker != 'User A':
        mismatches += 1
    print(f"  {match} Turn {turn_id} (Speaker: {speaker}): '{r['ambiguous_phrase']}'")

if mismatches > 0:
    print(f'\n⚠️  WARNING: {mismatches} User A resolutions are on User B turns!')
    print('This means we\'re trying to resolve User B\'s phrases with User A\'s context!')
print()

print('User B resolutions (first 5):')
for r in user_b_gt[:5]:
    turn_id = r['trigger_turn_id']
    turn = next((t for t in data['conversation_transcript'] if t['turn_id'] == turn_id), None)
    speaker = turn['speaker'] if turn else 'UNKNOWN'
    match = '✓' if speaker == 'User B' else '✗ MISMATCH'
    print(f"  {match} Turn {turn_id} (Speaker: {speaker}): '{r['ambiguous_phrase']}'")
print()

# Check filtered GT (what we actually evaluate)
print('='*80)
print('FILTERED GT (spatial/temporal only):')
print('='*80)

spatial_temporal_phrases = ['here', 'there', 'this', 'that', 'tomorrow', 'today', 'then', 
                            'later', 'soon', 'next', 'now', 'upstairs', 'downstairs',
                            'building', 'room', 'floor', 'office', 'location', 'place']

user_a_filtered = [r for r in user_a_gt if any(phrase in r.get('ambiguous_phrase', '').lower() 
                                                 for phrase in spatial_temporal_phrases)]

print(f'User A spatial/temporal GT: {len(user_a_filtered)}')
print()

print('Checking speaker match for filtered GT:')
mismatches_filtered = 0
for r in user_a_filtered:
    turn_id = r['trigger_turn_id']
    turn = next((t for t in data['conversation_transcript'] if t['turn_id'] == turn_id), None)
    speaker = turn['speaker'] if turn else 'UNKNOWN'
    match = '✓' if speaker == 'User A' else '✗ MISMATCH'
    if speaker != 'User A':
        mismatches_filtered += 1
    print(f"  {match} Turn {turn_id} (Speaker: {speaker}): '{r['ambiguous_phrase']}' → '{r['resolved_entity']}'")

print()
if mismatches_filtered > 0:
    print(f'❌ PROBLEM FOUND: {mismatches_filtered}/{len(user_a_filtered)} filtered GT are on wrong speaker!')
    print('   This is lowering BLEU because we evaluate User B phrases with User A context!')
else:
    print(f'✅ All {len(user_a_filtered)} filtered GT match User A (correct speaker)')
