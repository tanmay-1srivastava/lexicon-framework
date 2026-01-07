import os
import json
from openai import AzureOpenAI

try: from secret_keys import Open_ai_key
except ImportError: Open_ai_key = "YOUR_KEY_HERE"

# --- Configuration ---
endpoint = "https://initial-resources.cognitiveservices.azure.com/"
model_name = "gpt-4.1"
deployment = "gpt-4.1"
subscription_key = Open_ai_key # Replace with your key
api_version = "2024-12-01-preview"

input_dir = "data_generation/new_data/generated_datasets"
output_dir = "data_generation/new_data/enriched_datasets"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

def enrich_dataset():
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    
    for filename in files:
        print(f"Processing {filename}...")
        
        # 1. Load Original Data
        with open(os.path.join(input_dir, filename), 'r') as f:
            data = json.load(f)
        
        transcript = data.get("conversation_transcript", [])
        context = data.get("mobile_context_snapshot", {})
        
        # 2. Prompt GPT to define the Relationship and Privacy Envelope
        prompt = f"""
        Analyze this medical conversation and mobile context.
        
        TRANSCRIPT:
        {json.dumps(transcript, indent=2)}
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        TASK:
        1. Identify the specific RELATIONSHIP between User A and User B (e.g., Spouse, Coworker, Friend, Specialist).
        2. Define a PRIVACY ENVELOPE: What specific clinical data points is User B allowed to know?
        3. Determine ANSWERING LOGIC: If a gap is identified, should the system answer it directly in front of User B, or ask User A for permission first?
        
        Return ONLY a JSON object with these keys:
        "identified_relationship": "...",
        "access_level": "Tier 1 (Full) / Tier 2 (Partial) / Tier 3 (Emergency Only)",
        "authorized_entities": ["medication_names", "appat_times", "etc"],
        "answering_protocol": "direct_answer / permission_required",
        "reasoning": "..."
        """

        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a medical data privacy officer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            enrichment_data = json.loads(response.choices[0].message.content)
            
            # 3. Append the new data to a NEW copy of the JSON
            enriched_data = data.copy()
            enriched_data["relationship_policy"] = enrichment_data
            
            # 4. Save to the new directory (No Overwriting)
            output_filename = filename.replace(".json", "_enriched.json")
            with open(os.path.join(output_dir, output_filename), 'w') as f:
                json.dump(enriched_data, f, indent=4)
                
            print(f"Successfully enriched and saved: {output_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    enrich_dataset()