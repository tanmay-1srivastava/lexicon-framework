"""
Baseline GPT Test - No Context Aggregation Framework
Direct GPT processing of transcripts without structured context aggregation
"""

import json
import os
import sys
from openai import AzureOpenAI
from secret_keys import Open_ai_key

class BaselineGPTProcessor:
    """Direct GPT processing without Context Aggregation framework"""
    
    def __init__(self):
        # Azure OpenAI configuration (same as your setup)
        self.endpoint = os.getenv("ENDPOINT_URL", "https://initial-resources.cognitiveservices.azure.com/")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
        self.subscription_key = Open_ai_key
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2025-01-01-preview",
        )
    
    def process_raw_transcript(self, transcript_file, context_file):
        """Process transcript with minimal prompting - no structured framework"""
        
        # Load files
        with open(transcript_file, 'r') as f:
            transcript = f.read().strip()
        
        with open(context_file, 'r') as f:
            context_data = json.load(f)
        
        # Simple prompt - just ask GPT to analyze everything at once
        prompt = f"""
        Transcript: {transcript}
        
        Context: {json.dumps(context_data, indent=2)}
        
        Please analyze this conversation and provide:
        1. What the conversation is about
        2. What action items were mentioned
        3. What information is missing
        4. How urgent this is
        5. What questions need to be answered
        """
        
        # Single GPT call without structured processing
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that analyzes conversations."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=600,
                temperature=0.1,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing transcript"

def run_baseline_comparison():
    """Run baseline GPT processing on all scenarios"""
    
    processor = BaselineGPTProcessor()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "context_aggregation_test")
    
    # Same scenarios as Context Aggregation test
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
    
    print("BASELINE GPT TEST - No Context Aggregation Framework")
    print("="*70)
    print("Testing how GPT performs without structured context processing")
    print("="*70)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'-'*50}")
        print(f"BASELINE TEST {i}: {scenario['name']}")
        print(f"{'-'*50}")
        
        try:
            # Check if files exist
            if not os.path.exists(scenario['transcript']):
                print(f"Error: {scenario['transcript']} not found")
                continue
            if not os.path.exists(scenario['context']):
                print(f"Error: {scenario['context']} not found") 
                continue
            
            # Process with baseline approach
            result = processor.process_raw_transcript(
                scenario['transcript'],
                scenario['context']
            )
            
            print("BASELINE GPT ANALYSIS:")
            print(result)
            
        except Exception as e:
            print(f"Error processing {scenario['name']}: {e}")
    
    print(f"\n{'='*70}")
    print("BASELINE TESTING COMPLETED")
    print("Compare these results with your Context Aggregation framework results!")
    print("="*70)

if __name__ == "__main__":
    run_baseline_comparison()