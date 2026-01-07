import os
import json
import numpy as np
from datetime import datetime
from openai import AzureOpenAI

# --- Use your existing Azure setup ---
try: from secret_keys import Open_ai_key
except ImportError: Open_ai_key = "YOUR_KEY_HERE"

class PersonaEvaluator:
    def __init__(self):
        self.endpoint = "https://initial-resources.cognitiveservices.azure.com/"
        self.deployment = "gpt-4.1"
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.endpoint,
            api_key=Open_ai_key,
        )

    def predict_one_sided(self, transcript, context):
        """
        Simulates live inference: Analyzes ONLY User A's signals 
        to predict the persona vector.
        """
        prompt = f"""
        Analyze User A's linguistic signals in this ONE-SIDED medical transcript.
        
        TRANSCRIPT: {json.dumps(transcript)}
        CONTEXT: {json.dumps(context)}
        
        TASK: Predict the relationship and privacy state. 
        Focus on:
        - Deictic Masking: Does User A say 'it' or 'that thing' to hide info?
        - Formality: Use of titles vs. fragments.
        - Trust: Use of 'we' or references to shared private spaces.

        Return ONLY JSON:
        {{
            "predicted_relationship": "Spouse/Friend/Coworker/Staff",
            "predicted_state": "Open/Closed",
            "predicted_tier": 1, 2, or 3
        }}
        """
        try:
            res = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "system", "content": "You are a Privacy Analyst."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(res.choices[0].message.content)
        except: return {}

    def run_eval(self, data_dir="data_generation/new_data/persona_enriched_datasets"):
        print(f"\n{'FILE':<25} | {'GT TIER':<8} | {'PRED TIER':<10} | {'STATE MATCH'}")
        print("-" * 65)

        stats = {"tier_correct": 0, "state_correct": 0, "total": 0}

        for filename in sorted(os.listdir(data_dir)):
            if not filename.endswith("_persona.json"): continue
            
            with open(os.path.join(data_dir, filename)) as f:
                data = json.load(f)

            # 1. Get Ground Truth (The persona_vector you appended earlier)
            gt_vector = data.get("persona_vector", {})
            gt_tier = gt_vector.get("access_tier")
            gt_state = gt_vector.get("privacy_state")

            # 2. Run Inference (The One-Sided Prediction)
            prediction = self.predict_one_sided(
                data['conversation_transcript'], 
                data['mobile_context_snapshot']
            )

            pred_tier = prediction.get("predicted_tier")
            pred_state = prediction.get("predicted_state")

            # 3. Compare
            tier_match = (gt_tier == pred_tier)
            state_match = (gt_state == pred_state)
            
            stats["total"] += 1
            if tier_match: stats["tier_correct"] += 1
            if state_match: stats["state_correct"] += 1

            status = "✓" if tier_match else "✗"
            print(f"{filename[:25]:<25} | {gt_tier:<8} | {pred_tier:<10} | {state_match}")

        # Final Report
        print("\n" + "="*40)
        print("ONE-SIDED EVALUATION RESULTS")
        print("="*40)
        print(f"Tier Accuracy:  {stats['tier_correct']/stats['total']:.1%}")
        print(f"State Accuracy: {stats['state_correct']/stats['total']:.1%}")
        print("="*40 + "\n")

if __name__ == "__main__":
    evaluator = PersonaEvaluator()
    evaluator.run_eval()