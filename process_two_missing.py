"""
Run Information Gap Detection and Context Resolution on Event Keywords Datasets
Processes all 100 files in data_generation/event_keywords/generated_datasets
"""

import json
import os
from pathlib import Path
from datetime import datetime
import sys

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'context_aggregation/new_methods'))

from improved_processor import ImprovedInfoGapProcessor, ImprovedResolutionProcessor

def process_dataset(file_path):
    """Process a single dataset file"""
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract data
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
        print(f"  {user_label}: Detecting info gaps...", end=' ', flush=True)
        detected_gaps = info_gap_proc.detect_gaps(
            user_transcript, 
            local_context, 
            user_resolutions
        )
        print(f"Found {len(detected_gaps)} gaps", flush=True)
        
        # 2. CONTEXT RESOLUTION (Spatial/Temporal)
        print(f"  {user_label}: Resolving references...", end=' ', flush=True)
        resolutions = resolution_proc.resolve_references(
            user_transcript,
            local_context,
            ref_time,
            backstory=backstory
        )
        print(f"Found {len(resolutions)} resolutions", flush=True)
        
        # Store results
        results[user_key] = {
            'user': user_label,
            'info_gaps': detected_gaps,
            'resolutions': resolutions,
            'transcript_turns': len(user_transcript)
        }
    
    return results

def main():
    """Process all event keywords datasets"""
    
    # Setup paths
    data_dir = Path("data_generation/event_keywords/generated_datasets")
    output_dir = Path("evaluation/event_keywords_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get only the 2 missing files
    dataset_files = [
        data_dir / "friends_meeting_005.json",
        data_dir / "friends_meeting_009.json"
    ]
    
    print("="*80)
    print("EVENT KEYWORDS DATASET ANALYSIS")
    print("="*80)
    print(f"Total datasets: {len(dataset_files)}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print()
    
    # Track statistics
    total_gaps = 0
    total_resolutions = 0
    processed_count = 0
    
    # Process each file
    for i, file_path in enumerate(dataset_files, 1):
        filename = file_path.name
        print(f"[{i}/{len(dataset_files)}] Processing {filename}...")
        
        try:
            # Process dataset
            results = process_dataset(file_path)
            
            # Save results
            output_path = output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Update statistics
            for user_key in ['user_a', 'user_b']:
                total_gaps += len(results[user_key]['info_gaps'])
                total_resolutions += len(results[user_key]['resolutions'])
            
            processed_count += 1
            print(f"  ✓ Saved to {output_path.name}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Print summary
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Datasets processed: {processed_count}/{len(dataset_files)}")
    print(f"Total info gaps detected: {total_gaps}")
    print(f"Total resolutions found: {total_resolutions}")
    print(f"Average gaps per dataset: {total_gaps/processed_count:.1f}")
    print(f"Average resolutions per dataset: {total_resolutions/processed_count:.1f}")
    print()
    print(f"Results saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
