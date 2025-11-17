#!/usr/bin/env python3
"""
Simple script to run Context Aggregation scenarios from parent directory
"""

import sys
import os

# Add the current directory to Python path
sys.path.append('.')

# Import your context aggregation framework
from context_aggregation import LLMClient, ContextAggregator

def run_single_scenario(transcript_file, context_file, scenario_name):
    """Run a single Context Aggregation scenario"""
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {scenario_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize components
        llm_client = LLMClient()
        aggregator = ContextAggregator(llm_client)
        
        # Process the scenario
        result = aggregator.process_transcript_file(transcript_file, context_file)
        
        # Display results
        print(f"\nORIGINAL TRANSCRIPT:")
        print(f"  {result.resolved_content['original_speech']}")
        
        print(f"\nRESOLVED SPEECH:")
        print(f"  {result.resolved_content['resolved_speech']}")
        
        print(f"\nEXTRACTED INFORMATION:")
        print(f"  Action Items: {result.resolved_content.get('action_items', 'None')}")
        print(f"  Topics: {result.resolved_content.get('topics', 'None')}")
        print(f"  Urgency: {result.resolved_content.get('urgency_level', 'Unknown')}")
        print(f"  Purpose: {result.resolved_content.get('main_purpose', 'Unknown')}")
        
        print(f"\nINFORMATION GAPS:")
        for i, gap in enumerate(result.information_gaps, 1):
            print(f"  {i}. {gap}")
        
        print(f"\nCONTEXT SUMMARY:")
        summary = result.context_summary
        print(f"  Type: {summary.get('conversation_type', 'Unknown')}")
        print(f"  Urgency: {summary.get('urgency_level', 'Unknown')}")
        print(f"  Collaboration Needed: {summary.get('requires_collaboration', 'Unknown')}")
        print(f"  Completeness Score: {summary.get('completeness_score', 0):.2f}")
        
        return result
        
    except Exception as e:
        print(f"Error processing {scenario_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all_scenarios():
    """Run all Context Aggregation scenarios"""
    
    print("Starting Context Aggregation Framework Testing")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "context_aggregation_test")
    
    # Define all scenarios
    scenarios = [
        {
            "name": "Meeting Scheduling",
            "transcript": os.path.join(test_dir, "transcript1_meeting_scheduling.txt"),
            "context": os.path.join(test_dir, "context1_meeting_scheduling.json")
        },
        {
            "name": "Urgent Issue Resolution", 
            "transcript": os.path.join(test_dir, "transcript2_urgent_issue.txt"),
            "context": os.path.join(test_dir, "context2_urgent_issue.json")
        },
        {
            "name": "Casual Coordination",
            "transcript": os.path.join(test_dir, "transcript3_casual_coordination.txt"), 
            "context": os.path.join(test_dir, "context3_casual_coordination.json")
        },
        {
            "name": "Emergency Response",
            "transcript": os.path.join(test_dir, "transcript4_emergency.txt"),
            "context": os.path.join(test_dir, "context4_emergency.json") 
        }
    ]
    
    results = []
    
    # Run each scenario
    for scenario in scenarios:
        # Check if files exist
        if not os.path.exists(scenario['transcript']):
            print(f"Error: {scenario['transcript']} not found")
            continue
        if not os.path.exists(scenario['context']):
            print(f"Error: {scenario['context']} not found") 
            continue
            
        result = run_single_scenario(
            scenario['transcript'],
            scenario['context'], 
            scenario['name']
        )
        
        if result:
            results.append({
                'name': scenario['name'],
                'result': result
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TESTING COMPLETED - {len(results)} scenarios processed")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    run_all_scenarios()