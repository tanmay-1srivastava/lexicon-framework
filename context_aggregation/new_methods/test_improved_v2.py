#!/usr/bin/env python3
"""Test improved conversational reference tracking"""

import json
from pathlib import Path
from improved_processor import ImprovedResolutionProcessor

DATA_FILE = Path('../../data_generation/new_data/generated_datasets/doctor_visit_002.json')
OUTPUT_FILE = Path('resolution_results/doctor_visit_002_resolution_v2.json')

def main():
    # Load dataset
    with open(DATA_FILE) as f:
        data = json.load(f)
    
    # ONE-SIDED: Only User A's messages
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A']
    context = data['mobile_context_snapshot']['user_a']
    backstory = data.get('backstory')
    ref_time = data.get('generation_metadata', {}).get('timestamp', '2024-11-17T09:30:00')
    
    print("Testing improved conversational reference tracking...")
    print(f"One-sided transcript: {len(transcript)} User A turns")
    
    # Run resolution
    processor = ImprovedResolutionProcessor()
    resolutions = processor.resolve_references(transcript, context, ref_time, backstory)
    
    # Build result
    result = {
        "dataset_id": data['dataset_id'],
        "file": "doctor_visit_002.json",
        "mode": "improved_v2_conversational",
        "ground_truth_resolutions": []
    }
    
    for res in resolutions:
        result["ground_truth_resolutions"].append({
            "turn_id": res['turn_id'],
            "ambiguous_phrase": res['ambiguous_phrase'],
            "resolved_entity": res['resolved_entity'],
            "resolution_type": res['resolution_type'],
            "metadata_source": res['metadata_source']
        })
    
    # Save
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nGenerated {len(result['ground_truth_resolutions'])} resolutions")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Show key examples
    print("\n=== KEY EXAMPLES ===")
    for res in result['ground_truth_resolutions']:
        if res['turn_id'] in [5, 17, 27, 51]:
            print(f"Turn {res['turn_id']}: '{res['ambiguous_phrase']}' → {res['resolved_entity']}")
            print(f"  Source: {res['metadata_source']}\n")

if __name__ == '__main__':
    main()
