#!/usr/bin/env python3
"""Run improved V3 with contextual descriptions"""

import json
from pathlib import Path
from improved_processor import ImprovedResolutionProcessor

DATA_DIR = Path('../../data_generation/new_data/generated_datasets')
OUTPUT_DIR = Path('resolution_results_v3')

DATASETS = [
    'doctor_visit_001.json',
    'doctor_visit_002.json', 
    'doctor_visit_003.json',
    'friends_meeting_001.json',
    'friends_meeting_002.json',
    'friends_meeting_003.json',
    'work_collaboration_001.json',
    'work_collaboration_002.json',
    'work_collaboration_003.json'
]

def process_dataset(data_file):
    print(f"\nProcessing {data_file.name}...")
    
    with open(data_file) as f:
        data = json.load(f)
    
    transcript = [t for t in data['conversation_transcript'] if t['speaker'] == 'User A']
    context = data['mobile_context_snapshot']['user_a']
    backstory = data.get('backstory')
    ref_time = data.get('generation_metadata', {}).get('timestamp', '2024-11-17T09:30:00')
    
    processor = ImprovedResolutionProcessor()
    resolutions = processor.resolve_references(transcript, context, ref_time, backstory)
    
    result = {
        "dataset_id": data['dataset_id'],
        "file": data_file.name,
        "mode": "improved_v3_contextual",
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
    
    output_file = OUTPUT_DIR / data_file.name.replace('.json', '_resolution_v3.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✓ {len(result['ground_truth_resolutions'])} resolutions")
    return len(result['ground_truth_resolutions'])

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("RUNNING V3: CONTEXTUAL DESCRIPTIONS")
    print("="*60)
    
    total = 0
    for dataset_name in DATASETS:
        data_file = DATA_DIR / dataset_name
        if data_file.exists():
            total += process_dataset(data_file)
    
    print(f"\n{'='*60}")
    print(f"Total: {total} resolutions")
    print(f"Saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == '__main__':
    main()
