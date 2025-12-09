"""
Batch Dataset Generator - Generate 9 datasets across 3 scenarios
"""
from generate_dataset_azure import generate_dataset
import time

# Define the 3 main scenarios, each with 3 variations
scenarios = {
    "friends_meeting": [
        "Two close friends meeting at a coffee shop to plan a surprise birthday party for their mutual friend while keeping it secret",
        "Two friends coordinating a weekend hiking trip with multiple location changes and equipment sharing",
        "Two friends planning a group dinner event while managing dietary restrictions and venue availability"
    ],
    "work_collaboration": [
        "Two colleagues coordinating a critical product launch with multiple stakeholders and tight deadlines",
        "Two team members managing a software deployment with configuration details and resource allocation",
        "Two project managers synchronizing tasks across departments with budget constraints and approvals"
    ],
    "doctor_visit": [
        "A doctor and patient discussing a complex treatment plan involving medication schedules and specialist referrals",
        "A physician coordinating with a patient about test results while managing privacy concerns and family involvement",
        "A doctor and patient planning a surgery procedure with pre-op requirements and recovery coordination"
    ]
}

def main():
    print("=" * 70)
    print("🎯 BATCH DATASET GENERATION")
    print("=" * 70)
    print(f"\nGenerating 9 datasets (3 per scenario category)")
    print("\nScenario Categories:")
    print("  1. Friends Meeting (3 variations)")
    print("  2. Work Collaboration (3 variations)")
    print("  3. Doctor Visit (3 variations)")
    print("\n" + "=" * 70 + "\n")
    
    all_results = []
    dataset_count = 0
    
    for category, scenario_list in scenarios.items():
        print(f"\n{'#' * 70}")
        print(f"# CATEGORY: {category.upper().replace('_', ' ')}")
        print(f"{'#' * 70}\n")
        
        for i, scenario in enumerate(scenario_list, 1):
            dataset_count += 1
            
            print(f"\n{'-' * 70}")
            print(f"Dataset {dataset_count}/9 - {category} variation {i}")
            print(f"{'-' * 70}")
            
            # Generate output filename
            output_file = f"generated_datasets/{category}_{i:03d}.json"
            
            try:
                # Generate the dataset
                dataset = generate_dataset(scenario, output_file)
                
                all_results.append({
                    'dataset_id': dataset_count,
                    'category': category,
                    'variation': i,
                    'status': 'success',
                    'output_file': output_file,
                    'generated_id': dataset.get('dataset_id'),
                    'conversation_turns': len(dataset.get('conversation_transcript', [])),
                    'resolutions': len(dataset.get('ground_truth_resolutions', [])),
                    'queries': len(dataset.get('required_protocol_queries', []))
                })
                
                print(f"✅ Successfully generated: {output_file}")
                
                # Small delay to avoid rate limiting
                if dataset_count < 9:
                    print("\n⏳ Waiting 2 seconds before next generation...")
                    time.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed to generate dataset: {str(e)}")
                all_results.append({
                    'dataset_id': dataset_count,
                    'category': category,
                    'variation': i,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failed_count = sum(1 for r in all_results if r['status'] == 'failed')
    
    print(f"\nTotal Datasets: {len(all_results)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    
    if success_count > 0:
        print("\n📁 Generated Files:")
        for result in all_results:
            if result['status'] == 'success':
                print(f"  • {result['output_file']}")
                print(f"    - Turns: {result['conversation_turns']}, " +
                      f"Resolutions: {result['resolutions']}, " +
                      f"Queries: {result['queries']}")
    
    if failed_count > 0:
        print("\n⚠️  Failed Datasets:")
        for result in all_results:
            if result['status'] == 'failed':
                print(f"  • {result['category']} variation {result['variation']}: {result['error']}")
    
    print("\n" + "=" * 70)
    print("✨ Batch generation complete!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
