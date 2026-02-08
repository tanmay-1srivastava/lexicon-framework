import json
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'context_aggregation/new_methods'))

from improved_processor import ImprovedInfoGapProcessor, ImprovedResolutionProcessor

files = [
    'data_generation/event_keywords/generated_datasets/friends_meeting_005.json',
    'data_generation/event_keywords/generated_datasets/friends_meeting_009.json'
]

for file in files:
    print(f'Processing {file}...')
    with open(file, 'r') as f:
        data = json.load(f)
    
    backstory = data.get('backstory', {})
    transcript = data['conversation_transcript']
    context_snapshot = data['mobile_context_snapshot']
    
    # Initialize processors
    info_gap_proc = ImprovedInfoGapProcessor()
    resolution_proc = ImprovedResolutionProcessor()
    
    results = {}
    
    # Process for both User A and User B
    for user_key in ['user_a', 'user_b']:
        user_label = "User A" if user_key == 'user_a' else "User B"
        
        # Filter transcript for this user
        user_transcript = [t for t in transcript if t['speaker'] == user_label]
        
        # Get context
        local_context = context_snapshot[user_key]
        
        # Get reference time from first message
        ref_time = datetime.now().isoformat()
        
        # Get ground truth resolutions if available
        known_resolutions = data.get('ground_truth_resolutions', [])
        user_resolutions = [r for r in known_resolutions 
                           if user_label in r.get('resolution_source', '')]
        
        # 1. INFORMATION GAP DETECTION
        detected_gaps = info_gap_proc.detect_gaps(
            user_transcript, 
            local_context, 
            user_resolutions
        )
        
        # 2. CONTEXT RESOLUTION (Spatial/Temporal)
        resolutions = resolution_proc.resolve_references(
            user_transcript,
            local_context,
            ref_time,
            backstory=backstory
        )
        
        # Store results
        results[user_key] = {
            'user': user_label,
            'info_gaps': detected_gaps,
            'resolutions': resolutions,
            'transcript_turns': len(user_transcript)
        }
    
    result = {
        'dataset_name': file.split('/')[-1].replace('.json', ''),
        'backstory': backstory,
        'user_a': results['user_a'],
        'user_b': results['user_b']
    }
    
    output_file = 'evaluation/event_keywords_results/' + file.split('/')[-1].replace('.json', '_results.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'✓ Saved to {output_file}')

print('\nDone! All 100 datasets processed.')
