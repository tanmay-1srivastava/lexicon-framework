import json
from openai import AzureOpenAI

try: from secret_keys import Open_ai_key
except ImportError: Open_ai_key = "YOUR_KEY_HERE"

class InfoGapProcessor:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="https://initial-resources.cognitiveservices.azure.com/",
            api_key=Open_ai_key,
            api_version="2024-12-01-preview"
        )

    def detect_gaps(self, transcript, local_context, known_resolutions):
        """Lexicon Framework: Multi-strategy gap detection with entity completeness and mandatory fields."""
        conv_text = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in transcript])
        
        prompt = f"""Task: Identify High-Value Information Gaps using COMPREHENSIVE DETECTION.

INTERNAL RESOLUTIONS (Your current high-confidence context matches):
{json.dumps(known_resolutions, indent=2)}

LOCAL CONTEXT: {local_context}
CONVERSATION: {conv_text}

===== DETECTION STRATEGIES =====

**STRATEGY 1: ENTITY COMPLETENESS CHECKER**
For each critical entity mentioned, verify ALL required attributes are present in the transcript:

MEDICATION → Required: [name, dosage, frequency, timing]
  Example: "your medication" → Missing: ALL attributes → FLAG GAP
  Example: "Methotrexate" → Missing: dosage, frequency, timing → FLAG GAP

APPOINTMENT/LAB → Required: [type, date, time, location, prep_instructions]
  Example: "bloodwork tomorrow" → Missing: time, location, prep → FLAG GAP
  Example: "bloodwork at the lab" → Missing: date, time, which lab → FLAG GAP

PERSON (Medical Staff) → Required: [full_name, title/role, contact_method, department]
  Example: "Kara at extension 822" → Missing: full name, title, department → FLAG GAP
  Example: "the specialist" → Missing: ALL attributes → FLAG GAP

SPECIALIST REFERRAL → Required: [name, specialty, contact, appointment_status, reason]
  Example: "Dr. Patel will call" → Missing: contact, when, what if no call → FLAG GAP

DOCUMENT → Required: [type, identifier/location, purpose, how_to_access]
  Example: "red folder" → Missing: what's inside, why needed → FLAG GAP

LOCATION → Required: [building, floor, room/area, directions]
  Example: "the lab" → Missing: which lab, floor, room → FLAG GAP

TIME/DATE → Required: [specific_date, specific_time, timezone/relative_context]
  Example: "tomorrow" → Missing: specific time → FLAG GAP
  Example: "later" → Missing: date and time → FLAG GAP

**STRATEGY 2: QUESTION DETECTION & ANSWER VERIFICATION**
Track when user asks a question - verify if it gets COMPLETELY answered:

Pattern: User asks → Did answer include ALL necessary details?
  Turn X: "Which folder?" → If answer is just "red" without context → FLAG GAP (explain what's in red folder)
  Turn X: "Is it private?" → If no answer at all → FLAG HIGH-PRIORITY GAP

**STRATEGY 3: EXPANDED DEICTIC TRACKING**
Flag ALL ambiguous references, not just pronouns:

TEMPORAL: "tomorrow", "later", "then", "next week", "soon", "after", "before"
SPATIAL: "there", "here", "upstairs", "downstairs", "across", "nearby"
DEMONSTRATIVE: "that", "this", "those", "these", "it", "they"
DEFINITE DESCRIPTIONS: "the medication", "the doctor", "the lab", "the appointment"

If ANY of these appear WITHOUT concrete resolution in transcript → FLAG GAP

**STRATEGY 4: CRITICAL PATH ANALYSIS**
Identify action-requiring statements and verify completeness:

ACTION KEYWORDS: "schedule", "call", "contact", "go to", "take", "bring", "upload", "check", "confirm"
  Example: "Upload a photo" → Need: which app? how? privacy? → FLAG GAP
  Example: "Take it in the morning" → Need: what is "it"? exact time? → FLAG GAP
  Example: "She'll call you" → Need: who? when? backup plan? → FLAG GAP

**SPECIAL RULE FOR CALLBACKS/TIME WINDOWS:**
When a callback or appointment has a time RANGE (e.g., "between noon and two", "sometime tomorrow"):
  → ALWAYS FLAG: "What should happen if the call/appointment doesn't occur by the end of the window?"
  → This is CRITICAL for patient action planning

**SPECIAL RULE FOR BODY LOCATIONS:**
When medical observations mention body parts or "injection site", "swelling", "pain", etc.:
  → ALWAYS FLAG: "Specify exact anatomical location (which arm/leg, left/right, specific area)"
  → Required for proper medical documentation

**STRATEGY 5: FOLLOW-UP CHAIN TRACKING**
Track entities referenced multiple times without explicit naming:

If same entity referenced 2+ times across turns without full details → FLAG HIGH-PRIORITY GAP
  Example:
    Turn 8: "your medication" (vague)
    Turn 10: "take it in morning" (still vague)
    Turn 12: "track after each dose" (STILL no name!)
    → FLAG: "Medication referenced 3 times without naming - what is it?"

**STRATEGY 6: MANDATORY FIELD EXTRACTION**
For medical conversations, ensure these fields are captured:

MANDATORY PATIENT INSTRUCTIONS:
□ Primary medication name + dosage + schedule
□ Next appointment (date + time + location)
□ Lab work details (what + where + when + prep)
□ Specialist contact (who + when + how to reach)
□ Follow-up actions with deadlines
□ Emergency contact procedures

**STRATEGY 7: CONTEXT CROSS-VALIDATION**
Compare mentions against available context - if context CAN'T fulfill reference → GAP

**STRATEGY 8: ROLE-BASED ACCESS & PRIVACY BOUNDARIES (TPR IMPROVEMENT)**
For non-medical relationships (Coworker, Friend, Spouse), treat privacy boundaries as Mandatory Entities:
- IDENTITY: Who exactly is "him", "my friend", or "the team"?
- PARTITION: What specific medical info is shared vs. restricted?
- CONSENT: Explicitly flag gaps where a third party is mentioned but their access level to the patient's record is undefined.
- PRIVACY seeking behavior (asking for a room, asking to talk alone) is ALWAYS high priority.

**STRATEGY 9: CONVERSATIONAL PREFERENCE DETECTION**
Flag unresolved preference/choice questions as gaps:

PREFERENCE KEYWORDS: "prefer", "want to", "should we", "would you like", "do you want", "which one"
  Example: "Should we wait for your husband?" → If no clear answer → FLAG GAP
  Example: "Which folder?" → If answer is vague → FLAG GAP (need specifics)
  Example: "Do you want to discuss privately?" → If ignored/no explicit yes/no → FLAG GAP

**STRATEGY 10: IMPLICIT CONFIRMATION DETECTION**
When user responds with minimal confirmations that don't repeat the entity:

MINIMAL CONFIRMATIONS: "yes", "okay", "that one", "that room", "there", "then", "sure", "right"
  → Look back 1-2 turns for what entity/action is being confirmed
  → If entity NOT fully specified in those turns → FLAG GAP
  
  Example:
    Turn 4: "Go to the consultation room"
    Turn 5: "Okay, I'll head there" 
    → If "consultation room" location not specified (floor/directions) → FLAG GAP

**STRATEGY 11: AGGRESSIVE PRONOUN RESOLUTION**
Flag ALL pronouns and person references without clear named antecedents:

PRONOUNS: "he", "she", "him", "her", "they", "them"
VAGUE PERSON REFS: "someone", "person", "coordinator", "assistant", "volunteer", "therapist", "nurse"

Check: Was this person explicitly named with full details in last 3 turns?
  If NO → FLAG IDENTITY GAP
  
  Example: "Michelle will help" → Need: Michelle's full name, role, department
  Example: "He prefers chocolate" → Need: Who is "he"?

**STRATEGY 12: UNCONFIRMED CRITICAL INFORMATION**
When critical info is stated but NOT explicitly confirmed by user:

CRITICAL INFO TYPES: appointments, medications, procedures, deadlines, locations, referrals
  → If stated by doctor/staff but NO confirmation from patient in next 1-2 turns → FLAG "Needs confirmation"
  
  Example:
    Turn 20: Doctor: "Bloodwork tomorrow at 8am"
    Turn 21: Patient: "Okay"  (vague - did they understand?)
    → FLAG: "Confirm patient understands: bloodwork tomorrow at 8am, location, prep instructions"

**FILTERING RULES:**
- Ignore: greetings, small talk, coffee, fountains, "thank you"
- Prioritize: Privacy/Access > Medications > Appointments > Persons > Locations > Documents
- Focus on: actionable medical information, not social pleasantries
- Be AGGRESSIVE: When in doubt, flag the gap - better to over-ask than under-ask

**OUTPUT FORMAT:**
Return JSON: {{ "queries": [ 
  {{ "turn_id": 8, "strategy": "entity_completeness", "entity_type": "medication", "question": "What is the full medication name, dosage, and schedule?", "reason": "Medication mentioned 3 times without complete details" }},
  {{ "turn_id": 14, "strategy": "privacy_boundary", "entity_type": "person", "question": "Which specific coworker is referenced, and what parts of the record are they authorized to see?", "reason": "Privacy boundary undefined for coworker relationship" }},
  {{ "turn_id": 3, "strategy": "conversational_preference", "entity_type": "choice", "question": "Should we wait for your husband or proceed now?", "reason": "Preference question not clearly answered" }},
  {{ "turn_id": 5, "strategy": "implicit_confirmation", "entity_type": "location", "question": "Which consultation room - need building, floor, room number?", "reason": "User confirmed 'that room' but location details not specified" }},
  {{ "turn_id": 9, "strategy": "pronoun_resolution", "entity_type": "person", "question": "Who is Michelle - full name, role, and contact?", "reason": "Person referenced without full identification" }},
  {{ "turn_id": 20, "strategy": "unconfirmed_info", "entity_type": "appointment", "question": "Confirm patient understands bloodwork tomorrow at 8am - location and prep?", "reason": "Critical appointment stated but not explicitly confirmed with details" }}
] }}

BE AGGRESSIVE: Generate queries for ALL potential gaps. It's better to ask too many questions than miss critical information."""
        
        return self._call_llm(prompt)

    def detect_baseline(self, transcript, local_context):
        conv_text = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in transcript])
        prompt = f"What questions should be asked to fill gaps in this conversation? Context: {local_context}. Conversation: {conv_text}. Return JSON with 'queries' list of objects."
        return self._call_llm(prompt)

    def _call_llm(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1", messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            return data.get('queries', [])
        except: return []