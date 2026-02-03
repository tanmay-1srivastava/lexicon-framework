import json
from openai import AzureOpenAI

# --- Configuration (Azure setup) ---
try: from secret_keys import Open_ai_key
except ImportError: Open_ai_key = "YOUR_KEY_HERE"

class FuzzyLogicMatrix:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="https://initial-resources.cognitiveservices.azure.com/",
            api_key=Open_ai_key,
            api_version="2024-12-01-preview"
        )
        # THE DETERMINISTIC 3x3 SOCIAL NORMS TABLE
        self.lookup_table = {
            1: {"Low": "REVEAL", "Mid": "REVEAL", "High": "REVEAL"},
            2: {"Low": "REVEAL", "Mid": "PERMISSION", "High": "SUPPRESS"},
            3: {"Low": "PERMISSION", "Mid": "SUPPRESS", "High": "SUPPRESS"}
        }

    def _get_social_context(self, gap_text, transcript):
        """The 'Fuzzy' part: LLM categorizes sensitivity and detects state."""
        prompt = f"""
        Analyze the current conversation and the information gap.
        
        GAP: {gap_text}
        TRANSCRIPT: {json.dumps(transcript[-5:])} # Last 5 turns for context
        
        TASK:
        1. Rate DATA SENSITIVITY (Low: Logistical/Time/Loc | Mid: Clinical/Meds | High: Personal/Diagnoses).
        2. Detect PRIVACY STATE (Closed: Patient is being vague/masking | Open: Patient is direct).
        
        Return JSON: {{"sensitivity": "Low|Mid|High", "state": "Open|Closed"}}
        """
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)

    def enforce_social_norms(self, gap_text, transcript, trust_tier):
        """The 'Solid' part: Logic combines AI classification with the hard table."""
        
        # 1. Get AI Classification
        context = self._get_social_context(gap_text, transcript)
        sensitivity = context['sensitivity']
        state = context['state']

        # 2. Apply Privacy State Modifier (The "State Penalty")
        # If State is Closed, shift sensitivity up one level
        if state == "Closed":
            if sensitivity == "Low": sensitivity = "Mid"
            elif sensitivity == "Mid": sensitivity = "High"

        # 3. Deterministic Lookup
        action = self.lookup_table[trust_tier].get(sensitivity, "SUPPRESS")
        
        return {
            "action": action,
            "sensitivity_level": sensitivity,
            "privacy_state": state,
            "reason": f"Social Matrix: Tier {trust_tier} vs {sensitivity} sensitivity ({state} state)"
        }