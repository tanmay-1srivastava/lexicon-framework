import json
from openai import AzureOpenAI

try: from secret_keys import Open_ai_key
except ImportError: Open_ai_key = "YOUR_KEY_HERE"

class GeneralContextProcessor:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="https://initial-resources.cognitiveservices.azure.com/",
            api_key=Open_ai_key,
            api_version="2024-12-01-preview"
        )

    def resolve_framework(self, transcript, context, ref_time):
        conv_text = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in transcript])
        
        # FEW-SHOT EXAMPLES: Teaching the "Lexicon" logic using your medical dataset
        examples = """
        GOLDEN EXAMPLE 1 (Local Spatial):
        CONTEXT: { "location_semantic": "Dr. Chen's Office, Evergreen Medical Center" }
        TRANSCRIPT: "Turn 1: User A: Are you here already?"
        RESOLUTION: { "turn_id": 1, "ambiguous_phrase": "here", "resolved_entity": "Dr. Chen's Office", "evidence": "Direct match with location_semantic" }

        GOLDEN EXAMPLE 2 (Local Entity):
        CONTEXT: { "calendar_next": "12:00 PM - Lee, Marcus (Treatment Discussion)" }
        TRANSCRIPT: "Turn 8: User A: When did you last take it?"
        RESOLUTION: { "turn_id": 8, "ambiguous_phrase": "it", "resolved_entity": "Methotrexate medication", "evidence": "Linked to medication context in Treatment Discussion calendar entry" }

        GOLDEN EXAMPLE 3 (Information Gap - Peer Source):
        CONTEXT: { "calendar_next": "None" }
        TRANSCRIPT: "Turn 40: User A: Any issues since then?"
        RESOLUTION: { "turn_id": 40, "ambiguous_phrase": "then", "resolved_entity": "REQUIRES_PEER_QUERY", "evidence": "Timestamp of last dose is not in User A's local context" }
        """

        prompt = f"""Task: Resolve all ambiguous phrases using the Lexicon Methodology.
        Only resolve phrases found in the CONVERSATION SNIPPET. 
        For every resolution, you MUST provide the exact 'turn_id' from the snippet.

        {examples}

        REFERENCE TIME: {ref_time}
        LOCAL CONTEXT: {context}
        CONVERSATION SNIPPET:
        {conv_text}

        Return a JSON object with a "resolutions" list. Ensure "turn_id" is an integer:
        {{ "resolutions": [ {{ "turn_id": 8, "ambiguous_phrase": "it", "resolved_entity": "...", "evidence": "..." }} ] }}"""
        
        return self._call_llm(prompt)

    def resolve_baseline(self, transcript, context):
        conv_text = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in transcript])
        prompt = f"""Identify and resolve ambiguous phrases (it, here, her, then) in this conversation:
        {conv_text}
        
        Using this context: {context}
        
        Return a JSON object with a "resolutions" list containing "turn_id", "ambiguous_phrase", and "resolved_entity"."""
        return self._call_llm(prompt)

    def _call_llm(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1", messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            # Normalizing keys for robustness
            raw_list = data.get('resolutions', []) or data.get('results', []) or data.get('ambiguous_references', [])
            normalized = []
            for item in raw_list:
                normalized.append({
                    "turn_id": item.get('turn_id') or item.get('turnID') or item.get('turn'),
                    "ambiguous_phrase": item.get('ambiguous_phrase') or item.get('phrase'),
                    "resolved_entity": item.get('resolved_entity') or item.get('resolution')
                })
            return normalized
        except: return []