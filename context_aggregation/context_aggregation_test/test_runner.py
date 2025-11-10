"""
Test runner for Context Aggregation with simulated data
Processes all transcript/context pairs and shows results
"""

import os
import sys
import sys
sys.path.append('.')  # Add current directory to path
from context_aggregation import LLMClient, ContextAggregator

def run_all_scenarios():
    """Run Context Aggregation on all simulated scenarios"""
    
    # Initialize components - your context_aggregation.py handles the API key
    llm_client = LLMClient()
    aggregator = ContextAggregator(llm_client)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define test scenarios with full paths
    scenarios = [
        {
            "name": "Meeting Scheduling",
            "transcript": os.path.join(script_dir, "transcript1_meeting_scheduling.txt"),
            "context": os.path.join(script_dir, "context1_meeting_scheduling.json")
        },
        {
            "name": "Urgent Issue Resolution", 
            "transcript": os.path.join(script_dir, "transcript2_urgent_issue.txt"),
            "context": os.path.join(script_dir, "context2_urgent_issue.json")
        },
        {
            "name": "Casual Coordination",
            "transcript": os.path.join(script_dir, "transcript3_casual_coordination.txt"), 
            "context": os.path.join(script_dir, "context3_casual_coordination.json")
        },
        {
            "name": "Emergency Response",
            "transcript": os.path.join(script_dir, "transcript4_emergency.txt"),
            "context": os.path.join(script_dir, "context4_emergency.json") 
        }
    ]
    
    # Process each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*80}")
        
        try:
            # Check if files exist
            if not os.path.exists(scenario['transcript']):
                print(f"Error: {scenario['transcript']} not found")
                continue
            if not os.path.exists(scenario['context']):
                print(f"Error: {scenario['context']} not found")
                continue
            
            # Process the scenario
            result = aggregator.process_transcript_file(
                scenario['transcript'], 
                scenario['context']
            )
            
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
            
            print(f"\nINFORMATION GAPS IDENTIFIED:")
            for j, gap in enumerate(result.information_gaps, 1):
                print(f"  {j}. {gap}")
            
            print(f"\nCONTEXT SUMMARY:")
            summary = result.context_summary
            print(f"  Type: {summary.get('conversation_type', 'Unknown')}")
            print(f"  Urgency: {summary.get('urgency_level', 'Unknown')}")
            print(f"  Collaboration Needed: {summary.get('requires_collaboration', 'Unknown')}")
            print(f"  Completeness Score: {summary.get('completeness_score', 0):.2f}")
            
        except Exception as e:
            print(f"Error processing scenario: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("ALL SCENARIOS COMPLETED")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_all_scenarios()