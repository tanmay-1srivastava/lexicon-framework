import os
import sys
import json
from datetime import datetime
from openai import AzureOpenAI

# Add parent directory to path to import secret_keys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from secret_keys import Open_ai_key

# Azure OpenAI Configuration (from your Azure AI Studio)
endpoint = "https://initial-resources.cognitiveservices.azure.com/"
model_name = "gpt-4.1"
deployment = "gpt-4.1"
api_version = "2024-12-01-preview"

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=Open_ai_key,
)

def load_prompt():
    """Load the system prompt from prompt.txt"""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    with open(prompt_path, 'r') as f:
        return f.read()

def generate_dataset(scenario_description, output_file=None, temperature=0.9):
    """
    Generate a high-density synthetic dataset for context-aware assistant protocol
    
    Args:
        scenario_description: Description of the scenario to generate (e.g., "Organizing a protest")
        output_file: Path to save the output JSON (optional)
        temperature: Temperature for generation (default 0.9 for creativity)
    
    Returns:
        Generated dataset as a dictionary
    """
    system_prompt = load_prompt()
    
    user_prompt = f"""Generate a complete high-density dataset for the following scenario:

Scenario: {scenario_description}

Remember:
- ~2,000 words conversation (15 mins dialogue)
- 15+ local context resolutions (here, there, him, her, etc.)
- 8+ inter-agent queries
- ONLY generate queries for ACTIONABLE or SENSITIVE information
- Include complete metadata (GPS, WiFi, Calendar)
- Output valid JSON following the exact format specified

Generate the complete JSON output now:"""

    print(f"🚀 Generating dataset for scenario: {scenario_description}")
    print(f"Using Azure OpenAI model: {model_name}")
    print("This may take 30-60 seconds...\n")
    
    try:
        # Generate response
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            temperature=temperature,
            max_completion_tokens=13107,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        content = response.choices[0].message.content
        dataset = json.loads(content)
        
        # Add generation metadata
        dataset['generation_metadata'] = {
            'model': model_name,
            'temperature': temperature,
            'generated_at': datetime.now().isoformat(),
            'scenario_request': scenario_description,
            'provider': 'Azure OpenAI',
            'endpoint': endpoint
        }
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(os.path.dirname(__file__), output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"✅ Dataset saved to: {output_path}")
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        print(f"   - Dataset ID: {dataset.get('dataset_id', 'N/A')}")
        print(f"   - Conversation turns: {len(dataset.get('conversation_transcript', []))}")
        print(f"   - Context resolutions: {len(dataset.get('ground_truth_resolutions', []))}")
        print(f"   - Protocol queries: {len(dataset.get('required_protocol_queries', []))}")
        print(f"   - Relationship: {dataset.get('backstory', {}).get('relationship', 'N/A')}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error generating dataset: {str(e)}")
        raise

def generate_batch(scenarios, output_dir="generated_datasets"):
    """
    Generate multiple datasets from a list of scenarios
    
    Args:
        scenarios: List of scenario descriptions
        output_dir: Directory to save all generated datasets
    """
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"Generating dataset {i}/{len(scenarios)}")
        print(f"{'='*60}\n")
        
        # Generate output filename
        scenario_slug = scenario.lower().replace(' ', '_')[:40]
        output_file = f"{output_dir}/{scenario_slug}_{i:03d}.json"
        
        try:
            dataset = generate_dataset(scenario, output_file)
            results.append({
                'scenario': scenario,
                'status': 'success',
                'output_file': output_file,
                'dataset_id': dataset.get('dataset_id')
            })
        except Exception as e:
            results.append({
                'scenario': scenario,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save batch summary
    summary_path = os.path.join(os.path.dirname(__file__), output_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Batch generation complete!")
    print(f"   - Total: {len(scenarios)}")
    print(f"   - Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"   - Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"   - Summary saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    # Example scenarios for testing
    example_scenarios = [
        "Organizing a protest march in downtown with security concerns",
        "Planning a surprise 50th birthday party while keeping it secret",
        "Managing a hospital emergency with multiple doctors coordinating",
        "Coordinating a corporate merger with confidential information",
        "Planning a wedding with venue changes and family dynamics"
    ]
    
    print("🎯 Context-Aware Assistant Protocol Dataset Generator")
    print("=" * 60)
    print(f"Using: Azure OpenAI - {model_name}")
    print(f"Endpoint: {endpoint}")
    print("=" * 60)
    print("\nOptions:")
    print("1. Generate single dataset (interactive)")
    print("2. Generate batch from example scenarios")
    print("3. Custom batch (provide scenarios)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        scenario = input("\nEnter scenario description: ").strip()
        if scenario:
            output_file = input("Output filename (press Enter for no save): ").strip()
            output_file = output_file if output_file else None
            dataset = generate_dataset(scenario, output_file)
            print("\n📄 Generated Dataset Preview:")
            print(json.dumps(dataset, indent=2)[:1000] + "...\n")
    
    elif choice == "2":
        print(f"\nGenerating {len(example_scenarios)} datasets...")
        generate_batch(example_scenarios)
    
    elif choice == "3":
        print("\nEnter scenarios (one per line, empty line to finish):")
        custom_scenarios = []
        while True:
            line = input().strip()
            if not line:
                break
            custom_scenarios.append(line)
        
        if custom_scenarios:
            generate_batch(custom_scenarios)
        else:
            print("No scenarios provided.")
    
    else:
        print("Invalid choice.")
