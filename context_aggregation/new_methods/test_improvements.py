#!/usr/bin/env python3
"""
Test the improved processor and compare with old version
"""

import json
import sys
sys.path.insert(0, '.')

from improved_processor import ImprovedInfoGapProcessor, ImprovedResolutionProcessor
from info_gap_processor import InfoGapProcessor

def test_info_gap_improvements():
    """Test improved info gap detection"""
    
    print('='*80)
    print('TESTING IMPROVED INFO GAP DETECTION')
    print('='*80)
    print()
    
    # Load test data
    with open('../../data_generation/new_data/generated_datasets/doctor_visit_001.json') as f:
        data = json.load(f)
    
    # Get User A data
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A'][:10]
    context = data['mobile_context_snapshot']['user_a']
    known_gt = [g for g in data['ground_truth_resolutions'] if 'User A' in g['resolution_source']][:5]
    
    print('Test Transcript (first 10 User A turns):')
    for t in transcript[:3]:
        print(f"  Turn {t['turn_id']}: {t['text'][:60]}...")
    print(f"  ... and {len(transcript)-3} more turns")
    print()
    
    # OLD VERSION
    print('-'*80)
    print('OLD VERSION (single prompt for all turns):')
    print('-'*80)
    old_proc = InfoGapProcessor()
    old_gaps = old_proc.detect_gaps(transcript, context, known_gt)
    print(f'  Gaps detected: {len(old_gaps)}')
    print(f'  Turns covered: {set([g.get("turn_id") for g in old_gaps])}')
    for gap in old_gaps[:3]:
        print(f'  Turn {gap.get("turn_id")}: {gap.get("question", "N/A")[:60]}...')
    print()
    
    # NEW VERSION
    print('-'*80)
    print('NEW VERSION (turn-by-turn + priority filtering):')
    print('-'*80)
    new_proc = ImprovedInfoGapProcessor()
    new_gaps = new_proc.detect_gaps(transcript, context, known_gt)
    print(f'  Gaps detected: {len(new_gaps)}')
    print(f'  Turns covered: {sorted(set([g.get("turn_id") for g in new_gaps]))}')
    print(f'  Priority breakdown:')
    for priority in ['CRITICAL', 'HIGH', 'MEDIUM']:
        count = len([g for g in new_gaps if g.get('priority') == priority])
        print(f'    {priority}: {count}')
    print()
    print('  Sample gaps:')
    for gap in new_gaps[:5]:
        pri = gap.get('priority', 'UNKNOWN')
        turn = gap.get('turn_id')
        q = gap.get('question', 'N/A')[:70]
        print(f'  [{pri}] Turn {turn}: {q}...')
    print()
    
    print('='*80)
    print('COMPARISON:')
    print('='*80)
    print(f'Old: {len(old_gaps)} gaps, New: {len(new_gaps)} gaps')
    print(f'Turn coverage improved: {len(set([g.get("turn_id") for g in old_gaps]))} → {len(set([g.get("turn_id") for g in new_gaps]))} turns')
    print()

def test_resolution_improvements():
    """Test improved spatial/temporal resolution"""
    
    print('='*80)
    print('TESTING IMPROVED SPATIAL/TEMPORAL RESOLUTION')
    print('='*80)
    print()
    
    # Load test data
    with open('../../data_generation/new_data/generated_datasets/doctor_visit_001.json') as f:
        data = json.load(f)
    
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A'][:15]
    context = data['mobile_context_snapshot']['user_a']
    ref_time = "2024-11-17T09:30:00"
    
    print(f'Test Context:')
    print(f'  Location: {context.get("location_semantic", "N/A")}')
    print(f'  Calendar: {context.get("calendar_next", "N/A")}')
    print(f'  Reference time: {ref_time}')
    print()
    
    # NEW RESOLUTION PROCESSOR
    proc = ImprovedResolutionProcessor()
    resolutions = proc.resolve_references(transcript, context, ref_time)
    
    print(f'Resolutions found: {len(resolutions)}')
    print()
    
    # Group by type
    spatial = [r for r in resolutions if r.get('resolution_type') == 'spatial']
    temporal = [r for r in resolutions if r.get('resolution_type') == 'temporal']
    
    print(f'SPATIAL RESOLUTIONS: {len(spatial)}')
    for r in spatial[:5]:
        print(f'  Turn {r["turn_id"]}: "{r["ambiguous_phrase"]}" → "{r["resolved_entity"]}"')
        print(f'    Source: {r.get("metadata_source", "unknown")}')
    print()
    
    print(f'TEMPORAL RESOLUTIONS: {len(temporal)}')
    for r in temporal[:5]:
        print(f'  Turn {r["turn_id"]}: "{r["ambiguous_phrase"]}" → "{r["resolved_entity"]}"')
        print(f'    Source: {r.get("metadata_source", "unknown")}')
    print()
    
    # Test BLEU scoring
    print('='*80)
    print('TESTING BLEU-LIKE SCORING:')
    print('='*80)
    
    test_cases = [
        ("Dr. Chen's Office, Evergreen Medical Center", "Dr. Chen's Office, Evergreen Medical Center, 3rd floor"),
        ("Dr. Chen's Office", "Dr. Chen's Office, Evergreen Medical Center, 3rd floor"),
        ("Evergreen Medical Center", "Dr. Chen's Office, Evergreen Medical Center, 3rd floor"),
        ("Room 301", "Dr. Chen's Office, Evergreen Medical Center, 3rd floor"),
        ("Tomorrow at 9am", "November 18, 2024 at 9:00 AM"),
        ("12:00 PM appointment", "12:00 PM - Lee, Marcus (Treatment Discussion)"),
    ]
    
    for predicted, ground_truth in test_cases:
        score = proc.calculate_bleu_like_score(predicted, ground_truth)
        status = "EXACT" if score == 1.0 else ("PARTIAL" if score == 0.5 else "NO MATCH")
        print(f'{status} (score={score:.1f}):')
        print(f'  Predicted: "{predicted}"')
        print(f'  Ground truth: "{ground_truth}"')
        print()

if __name__ == "__main__":
    print('\\n' * 2)
    print('🚀 RUNNING IMPROVED PROCESSOR TESTS')
    print('='*80)
    print()
    
    try:
        test_info_gap_improvements()
        print('\\n' * 2)
        test_resolution_improvements()
        
        print('\\n' * 2)
        print('='*80)
        print('✅ ALL TESTS COMPLETED!')
        print('='*80)
        print()
        print('Next step: Run full evaluation to measure TPR improvement')
        print('Expected: 65.7% → 78%+ TPR')
        print()
        
    except Exception as e:
        print(f'\\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
