#!/usr/bin/env python3
"""Run V3 on 4 common files only for quick evaluation"""

import json
from pathlib import Path
from improved_processor import ImprovedResolutionProcessor

DATA_DIR = Path('../../data_generation/new_data/generated_datasets')
OUTPUT_DIR = Path('resolution_results_v3')

# Only the 4 common files
DATASETS = [
    'doctor_visit_001.json',
    'doctor_visit_002.json', 
    'doctor_visit_003.json',
    'friends_meeting_001.json'
]

def process_dataset(data_file):
    print(f"Processing {data_file.name}...")
    
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
        "ground_truth_resolutions": [
            {
                "turn_id": res['turn_id'],
                "ambiguous_phrase": res['ambiguous_phrase'],
                "resolved_entity": res['resolved_entity'],
                "resolution_type": res['resolution_type'],
                "metadata_source": res['metadata_source']
            }
            for res in resolutions
        ]
    }
    
    output_file = OUTPUT_DIR / data_file.name.replace('.json', '_resolution_v3.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✓ {len(result['ground_truth_resolutions'])} resolutions\n")
    return len(result['ground_truth_resolutions'])

OUTPUT_DIR.mkdir(exist_ok=True)

print("="*60)
print("V3: CONTEXTUAL DESCRIPTIONS (4 files)")
print("="*60 + "\n")

total = 0
for dataset_name in DATASETS:
    data_file = DATA_DIR / dataset_name
    if data_file.exists():
        total += process_dataset(data_file)

print("="*60)
print(f"Total: {total} resolutions in {OUTPUT_DIR}")
print("="*60)
