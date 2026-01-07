import os
import json
from openai import AzureOpenAI
try: from secret_keys import Open_ai_key
except ImportError: Open_ai_key = "YOUR_KEY_HERE"

# --- Configuration (Use your existing Azure details) ---
endpoint = "https://initial-resources.cognitiveservices.azure.com/"
deployment = "gpt-4.1"
subscription_key = Open_ai_key
api_version = "2024-12-01-preview"

input_dir = "data_generation/new_data/generated_datasets"
output_dir = "data_generation/new_data/persona_enriched_datasets"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

def analyze_persona_vector(transcript, context):
    """
    Analyzes one-sided conversation signals to build a Persona Vector.
    """
    prompt = f"""
    Analyze the following one-sided medical conversation and context.
    
    TRANSCRIPT: {json.dumps(transcript)}
    CONTEXT: {json.dumps(context)}
    
    TASK: Build a Multi-Dimensional Persona Vector for User B based on User A's linguistic signals.
    
    REQUIRED AXES (Score 1-10):
    1. FORMALITY: High (Technical terms, titles) vs Low (Slang, fragments).
    2. TRUST/PROXIMITY: High (Shared home context, 'we' language) vs Low (Distance, directions).
    3. OPENNESS STATE: 'Open' (Freely discussing symptoms) vs 'Closed' (Vague, masking 'it', seeking privacy).
    
    RELATIONSHIP IDENTIFICATION:
    - Based on the axes, identify the most likely role (Spouse, Friend, Coworker, Staff).

    Return ONLY a JSON object:
    {{
        "relationship": "...",
        "formality_score": 0,
        "trust_score": 0,
        "privacy_state": "Open/Closed",
        "access_tier": 1, 2, or 3,
        "linguistic_evidence": "Short quote or reason why",
        "answering_protocol": "direct_answer | permission_required | suppress"
    }}
    """
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": "You are a Social-Linguistic Analyst."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)

def enrich_with_persona():
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    
    for filename in files:
        print(f"Profiling Persona for {filename}...")
        with open(os.path.join(input_dir, filename), 'r') as f:
            data = json.load(f)
        
        # Run the Persona Analysis
        persona_vector = analyze_persona_vector(
            data.get("conversation_transcript", []),
            data.get("mobile_context_snapshot", {})
        )
        
        # Append to a NEW copy of the data
        enriched_data = data.copy()
        enriched_data["persona_vector"] = persona_vector
        
        # Logic Gate: Override access tier if 'Closed' state is detected
        if persona_vector["privacy_state"] == "Closed":
            enriched_data["persona_vector"]["access_tier"] = 3
            enriched_data["persona_vector"]["answering_protocol"] = "suppress"

        # Save new file
        output_name = filename.replace(".json", "_persona.json")
        with open(os.path.join(output_dir, output_name), 'w') as f:
            json.dump(enriched_data, f, indent=4)
        print(f"Saved: {output_name}")

if __name__ == "__main__":
    enrich_with_persona()