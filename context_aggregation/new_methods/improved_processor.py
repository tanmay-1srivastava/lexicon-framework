import json
import re
from openai import AzureOpenAI

try: from secret_keys import Open_ai_key
except ImportError: Open_ai_key = "YOUR_KEY_HERE"

class ImprovedInfoGapProcessor:
    """Improved processor with turn-by-turn analysis and better temporal handling"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="https://initial-resources.cognitiveservices.azure.com/",
            api_key=Open_ai_key,
            api_version="2024-12-01-preview"
        )

    def detect_gaps(self, transcript, local_context, known_resolutions):
        """IMPROVED: Turn-by-turn gap detection with priority filtering"""
        
        all_gaps = []
        
        # PROCESS EACH TURN INDIVIDUALLY
        for i, turn in enumerate(transcript):
            # Get context window (current + previous 3 turns)
            start_idx = max(0, i - 3)
            context_window = transcript[start_idx:i+1]
            
            # Analyze this specific turn
            turn_gaps = self._analyze_single_turn(turn, context_window, local_context, known_resolutions)
            all_gaps.extend(turn_gaps)
        
        # PRIORITY FILTERING - only return HIGH and CRITICAL
        prioritized_gaps = self._prioritize_gaps(all_gaps)
        high_priority = [g for g in prioritized_gaps if g.get('priority') in ['HIGH', 'CRITICAL']]
        
        return high_priority

    def _analyze_single_turn(self, turn, context_window, local_context, known_resolutions):
        """Analyze a single turn for information gaps"""
        
        conv_text = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" 
                               for t in context_window])
        
        # ENHANCED PROMPT WITH TECHNICAL TEMPLATES
        prompt = f"""Analyze Turn {turn['turn_id']} for information gaps.

CURRENT TURN TO ANALYZE:
Turn {turn['turn_id']}: {turn['speaker']}: {turn['text']}

PREVIOUS CONTEXT:
{conv_text}

LOCAL METADATA: {json.dumps(local_context, indent=2)}

KNOWN RESOLUTIONS: {json.dumps(known_resolutions[:5], indent=2)}

===== ENTITY TEMPLATES =====

MEDICAL ENTITIES:
- MEDICATION → [name, dosage, frequency, timing]
- APPOINTMENT → [type, date, time, location, prep_instructions]
- PERSON (Medical) → [full_name, title, contact, department]
- LAB/TEST → [type, date, time, location, prep, results_timeline]

TECHNICAL ENTITIES:
- CONFIG_FILE → [label/version, location, purpose, format]
- DIRECTORY/FOLDER → [full_path, contents, permissions, purpose]
- HARDWARE/EQUIPMENT → [type, description, location, specifications]
- CREDENTIALS → [system, username, key_location, access_level, expiry]
- DEPLOYMENT/TASK → [system, deadline, success_criteria, rollback_plan]

TEMPORAL ENTITIES (CRITICAL - USE CALENDAR!):
- "tomorrow" → Resolve to EXACT DATE using reference time
- "then", "later", "after" → Resolve to SPECIFIC TIME from calendar/context
- "next week", "soon" → Resolve to DATE RANGE with deadline
- Time windows → ALWAYS flag: "What if event doesn't happen by deadline?"

SPATIAL ENTITIES (USE LOCATION METADATA!):
- "here", "there" → Resolve using location_semantic, GPS, wifi_ssid
- "this room", "that building" → Resolve to BUILDING + FLOOR + ROOM
- "upstairs", "downstairs" → Resolve to FLOOR NUMBER

===== DETECTION STRATEGIES =====

1. ENTITY COMPLETENESS: Check if ALL required attributes are present
2. PRONOUN RESOLUTION: Flag "it", "he", "she", "they", "them" without clear antecedent
3. DEICTIC TRACKING: Flag "this", "that", "here", "there", "then" without resolution
4. PRIVACY/ACCESS: Flag when third party mentioned but access level undefined
5. CONFIRMATION NEEDED: Flag minimal confirmations ("okay", "yes") without explicit detail
6. UNANSWERED QUESTIONS: Track if questions get COMPLETE answers

===== FILTERING RULES =====

- FOCUS ONLY on turn {turn['turn_id']}
- Ignore greetings, "thank you", casual pleasantries
- Prioritize: Privacy > Medical > Technical > Appointments > Locations
- Return ONLY gaps where information is truly MISSING or AMBIGUOUS

OUTPUT: Return JSON with "queries" array. Each query MUST have turn_id = {turn['turn_id']}.
{{
  "queries": [
    {{
      "turn_id": {turn['turn_id']},
      "entity_type": "medication|appointment|location|temporal|person|config|credential",
      "ambiguous_phrase": "the specific vague phrase",
      "question": "What specific information is missing?",
      "priority": "CRITICAL|HIGH|MEDIUM",
      "reason": "Why this gap matters"
    }}
  ]
}}

BE SPECIFIC: Only flag gaps on turn {turn['turn_id']}. Do not generate questions for other turns.
"""
        
        return self._call_llm(prompt)

    def _prioritize_gaps(self, gaps):
        """Assign priority scores to gaps"""
        
        for gap in gaps:
            question = gap.get('question', '').lower()
            entity_type = gap.get('entity_type', '').lower()
            
            # CRITICAL: Privacy, medication, credentials, appointments
            if any(word in question for word in 
                   ['privacy', 'private', 'access', 'permission', 'who can see']):
                gap['priority'] = 'CRITICAL'
            elif any(word in question for word in 
                     ['medication', 'dose', 'prescription', 'allergy']):
                gap['priority'] = 'CRITICAL'
            elif any(word in question for word in 
                     ['password', 'credential', 'key', 'vault', 'auth']):
                gap['priority'] = 'CRITICAL'
            elif entity_type in ['appointment', 'lab', 'surgery']:
                gap['priority'] = 'CRITICAL'
            
            # HIGH: Person identity, specific locations, deadlines, technical specs
            elif any(word in question for word in 
                     ['who is', 'which person', 'contact', 'reach', 'name']):
                gap['priority'] = 'HIGH'
            elif any(word in question for word in 
                     ['where', 'location', 'building', 'floor', 'room']):
                gap['priority'] = 'HIGH'
            elif any(word in question for word in 
                     ['when', 'deadline', 'time', 'date', 'scheduled']):
                gap['priority'] = 'HIGH'
            elif any(word in question for word in 
                     ['config', 'version', 'label', 'specification']):
                gap['priority'] = 'HIGH'
            
            # MEDIUM: Everything else
            else:
                gap['priority'] = 'MEDIUM'
        
        return gaps

    def detect_baseline(self, transcript, local_context):
        """Baseline gap detection (for comparison)"""
        conv_text = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" 
                               for t in transcript])
        prompt = f"""What questions should be asked to fill gaps in this conversation?

Context: {local_context}
Conversation: {conv_text}

Return JSON with 'queries' list of objects containing 'question' and 'turn_id'."""
        
        return self._call_llm(prompt)

    def _call_llm(self, prompt):
        """Call LLM with error handling"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            return data.get('queries', [])
        except Exception as e:
            print(f"LLM error: {e}")
            return []


class ImprovedResolutionProcessor:
    """Improved spatial/temporal resolution with better scoring"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="https://initial-resources.cognitiveservices.azure.com/",
            api_key=Open_ai_key,
            api_version="2024-12-01-preview"
        )
    
    def resolve_references(self, transcript, context, ref_time, backstory=None):
        """Resolve ONLY spatial and temporal references using metadata - ONE-SIDED ANALYSIS"""
        
        # IMPORTANT: Only User A's turns - no info from other user
        conv_text = "\n".join([f"Turn {t['turn_id']}: {t['text']}" 
                               for t in transcript])
        
        # Include backstory for better context understanding (CoT insight)
        backstory_text = ""
        if backstory:
            summary = backstory.get('summary', '')
            relationship = backstory.get('relationship', '')
            backstory_text = f"""
SCENARIO CONTEXT:
- Summary: {summary}
- Relationship: {relationship}

This helps you understand WHY certain locations/times matter to User A.
"""
        
        prompt = f"""Resolve spatial and temporal ambiguous phrases using BOTH conversation context AND metadata.
{backstory_text}
CRITICAL: PRIORITIZE CONVERSATION CONTEXT OVER METADATA!

RESOLUTION PRIORITY (FOLLOW THIS ORDER):
1. CONVERSATION REFERENCES: If the phrase refers to something just mentioned in conversation, resolve to THAT (not GPS)
2. METADATA: Only use GPS/calendar if no clear conversational reference exists
3. CONTEXTUAL DESCRIPTIONS: Provide meaningful descriptions, not raw timestamps/coordinates

EXAMPLES OF CORRECT RESOLUTION:

Turn 4: "we can use that room next to the pharmacy"
Turn 5: "I'll head over there now"
→ "there" = "Consultation Room next to the pharmacy" (CONVERSATION REFERENCE)
→ NOT "Hospital Lounge" (current GPS location) ❌

Turn 22: "we can fit you in at 4:00 PM"
Turn 23: "Let's do today"
→ "today" = "Current date (message sent date)" (CONTEXTUAL)
→ NOT just "2024-11-17" (raw date) ❌

Turn 27: "I'll mark a slot for you at 4:00"
Turn 29: "Should I bring that medication then?"
→ "then" = "Scheduled follow-up appointment time (e.g., 4:00 PM)" (CONTEXTUAL)
→ NOT just "2024-11-17 14:00:00" (raw timestamp) ❌

ONE-SIDED ANALYSIS:
- You only see User A's messages (the speaker)
- You do NOT know what the other person said
- Infer from User A's responses what they're referring to

REFERENCE TIME: {ref_time}

METADATA AVAILABLE:
{json.dumps(context, indent=2)}

USER A MESSAGES ONLY:
{conv_text}

SPATIAL RESOLUTION RULES:
1. Check if User A is responding to/referring to a location mentioned in their PREVIOUS message
2. If User A says "I'll go there" - what location did they just acknowledge?
3. Provide DESCRIPTIVE locations with context, NOT just GPS names:
   - BAD: "Hospital Lounge"
   - GOOD: "Consultation Room next to the pharmacy"
   - BETTER: "Consultation Room next to the pharmacy (where the follow-up is scheduled)"
4. Only use location_semantic as LAST RESORT if:
   - "here" clearly refers to current location (e.g., "I'm here now")
   - No clear conversational reference exists
5. Add context to make locations meaningful:
   - Instead of: "Exam Room 3"
   - Use: "Exam Room 3 (for the follow-up appointment)"

TEMPORAL RESOLUTION RULES:
1. Provide CONTEXTUAL EVENT descriptions, NOT raw timestamps:
   - "now" → "Current time (message sent time)" NOT "2024-11-17 10:00:00"
   - "tomorrow" → "Next day from conversation date" NOT "2024-11-18"
   - "then" → "Scheduled follow-up appointment time (e.g., 2:00 PM or as agreed)" NOT "2024-11-17 14:00:00"
   - "later" → "After the current appointment/session" NOT a timestamp
2. For calendar events, describe the EVENT not the time:
   - BAD: "2024-11-17 14:00:00"
   - GOOD: "Scheduled follow-up appointment time (e.g., 2:00 PM or as agreed in this conversation)"
3. NEVER output raw timestamps or dates alone - always add context

CONVERSATION REFERENCE TRACKING:
- "there" after mentioning a place → that place (NOT GPS)
- "then" after mentioning a time → that time (NOT generic timestamp)
- "that" after discussing something → the discussed item
- Track what User A is acknowledging/responding to

OUTPUT: Return JSON with "resolutions" array:
{{
  "resolutions": [
    {{
      "turn_id": 5,
      "ambiguous_phrase": "there",
      "resolved_entity": "Consultation Room next to the pharmacy",
      "resolution_type": "spatial",
      "metadata_source": "Conversation Context"
    }},
    {{
      "turn_id": 27,
      "ambiguous_phrase": "then",
      "resolved_entity": "Scheduled follow-up appointment time (e.g., 2:00 PM or as agreed in this conversation)",
      "resolution_type": "temporal",
      "metadata_source": "User A Calendar"
    }},
    {{
      "turn_id": 7,
      "ambiguous_phrase": "now",
      "resolved_entity": "Current time (message sent time)",
      "resolution_type": "temporal",
      "metadata_source": "Conversation Context"
    }}
  ]
}}

CRITICAL REMINDERS:
- DESCRIBE the event/place, don't just give raw data
- Add context in parentheses when helpful
- Make resolutions human-readable and meaningful
- CONVERSATION CONTEXT > GPS/timestamps
"""
        
        return self._call_llm(prompt)
    
    def calculate_bleu_like_score(self, predicted, ground_truth):
        """Calculate improved fuzzy matching score for resolution
        
        Uses F1 token overlap with granular scoring to better reward partial matches.
        """
        
        # Tokenize (split into words, lowercase, remove punctuation)
        def tokenize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            return set(text.split())
        
        pred_tokens = tokenize(predicted)
        gt_tokens = tokenize(ground_truth)
        
        if not gt_tokens:
            return 0.0
        
        # Calculate precision (what fraction of predicted words are in ground truth)
        matches = pred_tokens.intersection(gt_tokens)
        precision = len(matches) / len(pred_tokens) if pred_tokens else 0.0
        
        # Calculate recall (what fraction of ground truth words are in prediction)
        recall = len(matches) / len(gt_tokens) if gt_tokens else 0.0
        
        # F1 score (harmonic mean of precision and recall)
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # Granular scoring - gives more credit for good partial matches
        if f1 >= 0.9:
            return 1.0  # Excellent match (90%+ overlap)
        elif f1 >= 0.7:
            return 0.9  # Very good match (70-90% overlap)
        elif f1 >= 0.5:
            return 0.7  # Good partial match (50-70% overlap)
        elif f1 >= 0.3:
            return 0.5  # Acceptable partial match (30-50% overlap)
        else:
            return 0.0  # Poor match (<30% overlap)
    
    def _call_llm(self, prompt):
        """Call LLM with error handling"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1  # Keep original temp
            )
            data = json.loads(response.choices[0].message.content)
            
            # Normalize resolutions
            resolutions = data.get('resolutions', [])
            normalized = []
            for item in resolutions:
                normalized.append({
                    "turn_id": item.get('turn_id'),
                    "ambiguous_phrase": item.get('ambiguous_phrase'),
                    "resolved_entity": item.get('resolved_entity'),
                    "resolution_type": item.get('resolution_type', 'unknown'),
                    "metadata_source": item.get('metadata_source', 'unknown')
                })
            return normalized
        except Exception as e:
            print(f"LLM error: {e}")
            return []
