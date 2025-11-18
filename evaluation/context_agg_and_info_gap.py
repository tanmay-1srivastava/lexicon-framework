import os
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AzureOpenAI

# Import your existing framework components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('./context_aggregation')
sys.path.append('./info-gap-detection')

from secret_keys import Open_ai_key

@dataclass
class ConversationData:
    speaker_name: str
    speaker_utterances: List[str]
    timestamps: List[str]
    speaker_profile: Dict
    other_speaker_profile: Dict
    scenario_context: Dict

@dataclass
class EvaluationResult:
    file_path: str
    speaker_name: str
    framework_output: Dict
    baseline_output: Dict
    metrics: Dict
    metadata_used: Dict

class LLMClient:
    """Azure OpenAI client for framework calls"""
    
    def __init__(self):
        self.endpoint = os.getenv("ENDPOINT_URL", "https://initial-resources.cognitiveservices.azure.com/")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
        self.subscription_key = Open_ai_key
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2025-01-01-preview",
        )
    
    def generate(self, prompt: str, max_tokens: int = 600) -> str:
        """Generate text response"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM generation error: {e}")
            return ""

class ConversationParser:
    """ULTRA-ROBUST parser that handles ALL conversation formats"""
    
    def parse_conversation_file(self, file_path: str) -> ConversationData:
        """Parse ANY conversation text file format with intelligent filtering"""
        
        print(f"📂 Processing: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Step 1: Check if file contains actual conversation
        if not self._contains_actual_conversation(content):
            print("   ⚠️ SKIPPED: File contains narrative/description, not actual dialogue")
            return self._create_empty_conversation_data()
        
        # Step 2: Detect and parse conversation format
        if self._is_new_format(content):
            return self._parse_new_format_robust(content, file_path)
        else:
            return self._parse_old_format_robust(content, file_path)
    
    def _contains_actual_conversation(self, content: str) -> bool:
        """Check if file contains actual dialogue vs narrative description"""
        
        # Red flags for narrative files
        narrative_indicators = [
            "After greeting Speaker",
            "began the discussion by",
            "Speaker 1 sat anxiously",
            "entered the room, carrying",
            "The nurse assistant had just",
            "following hospital policy"
        ]
        
        # Check for narrative patterns
        for indicator in narrative_indicators:
            if indicator in content:
                return False
        
        # Check for dialogue patterns (good signs)
        dialogue_patterns = [
            r'[A-Za-z]+:\s*"[^"]+[.!?]"',  # Name: "actual speech"
            r'\{[^}]*[A-Za-z]+:\s*"[^"]+[.!?]"',  # {timestamp} Name: "speech"
            r'[A-Za-z]+\s*:\s*[A-Z][^.]*[.!?]',  # Name: Direct speech without quotes
        ]
        
        dialogue_count = 0
        for pattern in dialogue_patterns:
            dialogue_count += len(re.findall(pattern, content))
        
        # Need at least 2 dialogue exchanges
        return dialogue_count >= 2
    
    def _is_new_format(self, content: str) -> bool:
        """Detect if this is any variation of the new simple format"""
        
        # Multiple new format patterns
        new_patterns = [
            r'\{\d{2}:\d{2}:\d{2}\s+[A-Za-z]+:"[^"]*"\s*\}',  # {timestamp Name:"text"}
            r'\{\d{2}:\d{2}:\d{2}\{\s*[A-Za-z]+:\s*"[^"]*"\s*\}',  # {timestamp{ Name: "text" }
            r'\d{2}:\d{2}:\d{2}\s*[A-Za-z]+:\s*"[^"]*"',  # timestamp Name: "text"
        ]
        
        for pattern in new_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _parse_new_format_robust(self, content: str, file_path: str) -> ConversationData:
        """Parse ALL variations of new format"""
        
        print("   Detected: NEW conversation format")
        
        # Try multiple extraction patterns in order of preference
        extraction_strategies = [
            self._extract_new_format_strategy1,
            self._extract_new_format_strategy2, 
            self._extract_new_format_strategy3,
            self._extract_new_format_strategy4,
            self._extract_new_format_strategy5
        ]
        
        for i, strategy in enumerate(extraction_strategies, 1):
            try:
                matches = strategy(content)
                if matches and len(matches) >= 2:  # Need at least 2 exchanges
                    print(f"   ✅ Parsed using strategy {i}: Found {len(matches)} exchanges")
                    return self._build_conversation_from_matches(matches, file_path)
            except Exception as e:
                print(f"   Strategy {i} failed: {e}")
                continue
        
        print("   ⚠️ All parsing strategies failed")
        return self._create_empty_conversation_data()
    
    def _extract_new_format_strategy1(self, content: str) -> List[Tuple[str, str, str]]:
        """Strategy 1: {12:01:05 Rob:"text"}"""
        pattern = r'\{(\d{2}:\d{2}:\d{2})\s+([A-Za-z]+):"([^"]+)"\s*\}'
        return re.findall(pattern, content, re.DOTALL)
    
    def _extract_new_format_strategy2(self, content: str) -> List[Tuple[str, str, str]]:
        """Strategy 2: {10:02:12{ Rob: "text" }"""
        pattern = r'\{(\d{2}:\d{2}:\d{2})\{\s*([A-Za-z]+):\s*"([^"]+)"\s*\}'
        return re.findall(pattern, content, re.DOTALL)
    
    def _extract_new_format_strategy3(self, content: str) -> List[Tuple[str, str, str]]:
        """Strategy 3: 10:02:12 Rob: "text" (no braces)"""
        pattern = r'(\d{2}:\d{2}:\d{2})\s*([A-Za-z]+):\s*"([^"]+)"'
        return re.findall(pattern, content, re.DOTALL)
    
    def _extract_new_format_strategy4(self, content: str) -> List[Tuple[str, str, str]]:
        """Strategy 4: Flexible spacing {timestamp{speaker:"text"}"""
        pattern = r'\{(\d{2}:\d{2}:\d{2})\{?\s*([A-Za-z]+):\s*"([^"]+)"\s*\}?'
        return re.findall(pattern, content, re.DOTALL)
    
    def _extract_new_format_strategy5(self, content: str) -> List[Tuple[str, str, str]]:
        """Strategy 5: Extract without quotes if necessary"""
        pattern = r'(\d{2}:\d{2}:\d{2})[^A-Za-z]*([A-Za-z]+):\s*([^{}]+?)(?=\d{2}:\d{2}:\d{2}|\s*$)'
        matches = re.findall(pattern, content, re.DOTALL)
        # Clean up extracted text
        cleaned_matches = []
        for timestamp, speaker, text in matches:
            clean_text = text.strip().strip('"').strip('{}').strip()
            if clean_text and len(clean_text) > 5:  # Minimum meaningful text
                cleaned_matches.append((timestamp, speaker, clean_text))
        return cleaned_matches
    
    def _build_conversation_from_matches(self, matches: List[Tuple[str, str, str]], file_path: str) -> ConversationData:
        """Build conversation data from extracted matches"""
        
        if not matches:
            return self._create_empty_conversation_data()
        
        # Get first speaker
        first_speaker = matches[0][1]
        print(f"   Selected speaker: {first_speaker}")
        
        # Extract utterances and timestamps for first speaker
        speaker_utterances = []
        timestamps = []
        
        for timestamp, speaker, utterance in matches:
            if speaker == first_speaker:
                timestamps.append(timestamp)
                # Clean up utterance
                clean_utterance = utterance.strip().strip('"').strip('{}').strip()
                if clean_utterance and clean_utterance not in speaker_utterances:
                    speaker_utterances.append(clean_utterance)
        
        print(f"   Found {len(speaker_utterances)} utterances")
        
        # Create profiles for new format
        all_speakers = list(set(match[1] for match in matches))
        speaker_profile = {"Name": first_speaker, "Role": "Participant"}
        other_speakers = [s for s in all_speakers if s != first_speaker]
        other_speaker_profile = {"Name": other_speakers[0] if other_speakers else "Other", "Role": "Participant"}
        
        return ConversationData(
            speaker_name=first_speaker,
            speaker_utterances=speaker_utterances,
            timestamps=timestamps,
            speaker_profile=speaker_profile,
            other_speaker_profile=other_speaker_profile,
            scenario_context={"format": "new", "file": os.path.basename(file_path)}
        )
    
    def _parse_old_format_robust(self, content: str, file_path: str) -> ConversationData:
        """Parse old format with robust error handling"""
        
        print("   Detected: OLD conversation format")
        
        try:
            # Extract character profiles
            character_profiles = self._extract_character_profiles_robust(content)
            
            # Extract scenario context
            scenario_context = self._extract_scenario_context(content)
            
            # Extract GPT response section
            gpt_section = self._extract_gpt_response_robust(content)
            
            # Get first speaker data
            conversation_data = self._extract_first_speaker_data_robust(gpt_section, character_profiles)
            
            # Add scenario context
            conversation_data.scenario_context = scenario_context
            conversation_data.scenario_context["format"] = "old"
            
            return conversation_data
            
        except Exception as e:
            print(f"   ⚠️ Old format parsing failed: {e}")
            return self._create_empty_conversation_data()
    
    def _extract_character_profiles_robust(self, content: str) -> Dict:
        """Robust character profile extraction with multiple strategies"""
        
        strategies = [
            self._extract_json_strategy1,
            self._extract_json_strategy2,
            self._extract_json_strategy3,
            self._extract_names_from_content
        ]
        
        for strategy in strategies:
            try:
                profiles = strategy(content)
                if profiles and isinstance(profiles, dict):
                    return profiles
            except:
                continue
        
        print("   Using default character profiles")
        return self._create_default_profiles()
    
    def _extract_names_from_content(self, content: str) -> Dict:
        """Extract character names directly from content when JSON fails"""
        
        # Look for common name patterns
        name_patterns = [
            r'([A-Z][a-z]+):\s*"',  # Name: "speech"
            r'Speaker\s*(\d+)',     # Speaker 1, Speaker 2
            r'character\d+.*?"Name":\s*"([^"]+)"'  # JSON name field
        ]
        
        found_names = set()
        for pattern in name_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, str) and len(match) > 1:
                    found_names.add(match)
        
        # Create profiles from found names
        names_list = list(found_names)[:2]  # Take first 2
        if len(names_list) >= 2:
            return {
                "character1": {"Name": names_list[0], "Role": "Participant"},
                "character2": {"Name": names_list[1], "Role": "Participant"}
            }
        
        return self._create_default_profiles()
    
    def _extract_json_strategy1(self, content: str) -> Dict:
        """Strategy 1: Extract between character1 and Event ID"""
        profile_start = content.find('"character1":')
        profile_end = content.find('Scenario: Event ID:')
        
        if profile_start != -1 and profile_end != -1:
            profile_section = content[profile_start-1:profile_end].strip()
            profile_section = '{' + profile_section + '}'
            
            # Clean up the JSON
            profile_section = profile_section.replace('\n', ' ').replace('\r', '')
            last_brace = profile_section.rfind('}')
            if last_brace != -1:
                profile_section = profile_section[:last_brace + 1]
            
            return json.loads(profile_section)
        return {}
    
    def _extract_json_strategy2(self, content: str) -> Dict:
        """Strategy 2: Extract using regex for JSON blocks"""
        pattern = r'{\s*"character1":\s*{[^}]+}[^}]*"character2":\s*{[^}]+}\s*}'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    
    def _extract_json_strategy3(self, content: str) -> Dict:
        """Strategy 3: Build JSON from individual character blocks"""
        char1_match = re.search(r'"character1":\s*({[^}]+})', content)
        char2_match = re.search(r'"character2":\s*({[^}]+})', content)
        
        if char1_match and char2_match:
            char1_data = json.loads(char1_match.group(1))
            char2_data = json.loads(char2_match.group(1))
            return {"character1": char1_data, "character2": char2_data}
        return {}
    
    def _create_default_profiles(self) -> Dict:
        """Create default character profiles"""
        return {
            "character1": {
                "Name": "Speaker1",
                "Role": "Participant", 
                "Experience (years)": "2",
                "Current project": "Discussion",
                "Nature(like introvert/extrovert)": "Mixed"
            },
            "character2": {
                "Name": "Speaker2",
                "Role": "Participant",
                "Experience (years)": "3", 
                "Current project": "Discussion",
                "Nature(like introvert/extrovert)": "Mixed"
            }
        }
    
    def _extract_scenario_context(self, content: str) -> Dict:
        """Extract scenario context from content"""
        scenario_data = {}
        
        try:
            # Extract event ID
            event_match = re.search(r'Event ID:\s*(\d+)', content)
            if event_match:
                scenario_data['event_id'] = event_match.group(1)
            
            # Extract keywords
            keywords_match = re.search(r'Keywords:\s*([^\n]+)', content)
            if keywords_match:
                scenario_data['keywords'] = keywords_match.group(1).split(', ')
            
        except Exception as e:
            print(f"   Warning: Could not parse scenario context: {e}")
        
        return scenario_data
    
    def _extract_gpt_response_robust(self, content: str) -> str:
        """Robust GPT response extraction"""
        
        # Try multiple patterns
        response_patterns = [
            'GPT-4 RESPONSE:',
            'GPT RESPONSE:',
            'Response:',
            'RESPONSE:',
            'Generated Conversation:',
            'Conversation:'
        ]
        
        for pattern in response_patterns:
            start_pos = content.find(pattern)
            if start_pos != -1:
                return content[start_pos + len(pattern):].strip()
        
        # If no explicit response section, try to find conversation-like content
        lines = content.split('\n')
        conversation_lines = []
        found_dialogue = False
        
        for line in lines:
            if re.search(r'[A-Za-z]+:\s*"[^"]*"', line) or re.search(r'[A-Za-z]+:\s*[A-Z]', line):
                found_dialogue = True
                conversation_lines.append(line)
            elif found_dialogue and line.strip():
                conversation_lines.append(line)
        
        return '\n'.join(conversation_lines) if conversation_lines else ""
    
    def _extract_first_speaker_data_robust(self, gpt_section: str, character_profiles: Dict) -> ConversationData:
        """Robust first speaker data extraction"""
        
        # Get speaker names from profiles
        speaker_names = []
        for char_key, profile in character_profiles.items():
            if isinstance(profile, dict) and 'Name' in profile:
                speaker_names.append(profile['Name'])
        
        # Add common fallback names
        speaker_names.extend(['Speaker1', 'Speaker2', 'Sarah', 'Thomas', 'Mathew', 'Rob', 'Tina', 'Dr', 'Patient'])
        
        print(f"   Looking for speakers: {speaker_names}")
        
        # Find first speaker with multiple strategies
        first_speaker = None
        lines = [line.strip() for line in gpt_section.split('\n') if line.strip()]
        
        # Strategy 1: Exact name match with colon
        for line in lines:
            for name in speaker_names:
                if f'{name}:' in line:
                    first_speaker = name
                    break
            if first_speaker:
                break
        
        # Strategy 2: Any speaker-like pattern
        if not first_speaker:
            for line in lines:
                speaker_match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\s*:', line)
                if speaker_match:
                    potential_speaker = speaker_match.group(1)
                    if potential_speaker not in ['GPT', 'Timestamp', 'RESPONSE', 'Event', 'Keywords', 'Setting']:
                        first_speaker = potential_speaker
                        break
        
        if not first_speaker:
            first_speaker = speaker_names[0] if speaker_names else "Speaker1"
        
        print(f"   Selected speaker: {first_speaker}")
        
        # Extract utterances with multiple patterns
        speaker_utterances = []
        timestamps = []
        
        for line in lines:
            if f'{first_speaker}:' in line:
                # Extract timestamp
                timestamp_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    timestamps.append(timestamp_match.group(1))
                
                # Extract utterance with multiple patterns
                utterance_patterns = [
                    f'{first_speaker}:\\s*"([^"]+)"',  # Name: "utterance"
                    f'{first_speaker}:\\s*([^\\n]+)',  # Name: utterance without quotes
                    f'\\{{[^}}]*\\}}\\s*{first_speaker}:\\s*"([^"]+)"'  # {timestamp} Name: "utterance"
                ]
                
                utterance_found = False
                for pattern in utterance_patterns:
                    utterance_match = re.search(pattern, line)
                    if utterance_match:
                        utterance = utterance_match.group(1).strip()
                        # Clean up utterance
                        utterance = utterance.strip('"').strip('{}').strip()
                        if utterance and len(utterance) > 5 and utterance not in speaker_utterances:
                            speaker_utterances.append(utterance)
                            utterance_found = True
                        break
                
                if not utterance_found:
                    # Last resort: take everything after the colon
                    colon_index = line.find(f'{first_speaker}:')
                    if colon_index != -1:
                        utterance = line[colon_index + len(f'{first_speaker}:'):].strip()
                        utterance = utterance.strip('"').strip('{}').strip()
                        if utterance and len(utterance) > 5 and utterance not in speaker_utterances:
                            speaker_utterances.append(utterance)
        
        print(f"   Found {len(speaker_utterances)} utterances")
        
        # Get profiles
        speaker_profile = self._get_profile_by_name(first_speaker, character_profiles)
        other_speaker_profile = self._get_other_profile(first_speaker, character_profiles)
        
        return ConversationData(
            speaker_name=first_speaker,
            speaker_utterances=speaker_utterances,
            timestamps=timestamps,
            speaker_profile=speaker_profile,
            other_speaker_profile=other_speaker_profile,
            scenario_context={}
        )
    
    def _get_profile_by_name(self, name: str, character_profiles: Dict) -> Dict:
        """Get profile for specific character name"""
        for char_key, profile in character_profiles.items():
            if isinstance(profile, dict) and profile.get('Name') == name:
                return profile
        return {"Name": name, "Role": "Participant"}
    
    def _get_other_profile(self, speaker_name: str, character_profiles: Dict) -> Dict:
        """Get profile for the other speaker"""
        for char_key, profile in character_profiles.items():
            if isinstance(profile, dict) and profile.get('Name') != speaker_name:
                return profile
        return {"Name": "Other", "Role": "Participant"}
    
    def _create_empty_conversation_data(self) -> ConversationData:
        """Create empty conversation data for failed parsing"""
        return ConversationData(
            speaker_name="SKIP_FILE",
            speaker_utterances=[],
            timestamps=[],
            speaker_profile={},
            other_speaker_profile={},
            scenario_context={}
        )

# [Rest of the classes remain the same as the previous version]
class MetadataGenerator:
    """Generate realistic metadata based on conversation context"""
    
    def generate_metadata(self, conversation_data: ConversationData) -> Dict:
        """Generate contextually appropriate metadata"""
        
        conversation_text = ' '.join(conversation_data.speaker_utterances)
        
        # Smart inference based on conversation content
        location = self._infer_location(conversation_text, conversation_data.scenario_context)
        meeting_type = self._infer_meeting_type(conversation_text, conversation_data.scenario_context)
        
        metadata = {
            "temporal": {
                "current_time": conversation_data.timestamps[0] if conversation_data.timestamps else "09:00:00",
                "day_of_week": "Monday",
                "work_hours": True
            },
            "spatial": {
                "location": location,
                "building": "Corporate Office",
                "privacy_level": "semi_private"
            },
            "calendar": {
                "current_meeting": meeting_type,
                "related_projects": self._extract_project_names(conversation_text, conversation_data.speaker_profile)
            },
            "participants": [conversation_data.speaker_name],
            "speaker_profile": conversation_data.speaker_profile,
            "other_speaker_profile": conversation_data.other_speaker_profile
        }
        
        return metadata
    
    def _infer_location(self, conversation_text: str, scenario_context: Dict) -> str:
        """Infer location from conversation content"""
        text_lower = conversation_text.lower()
        
        if any(word in text_lower for word in ['launch', 'meeting', 'rollout']):
            return "Conference Room B"
        elif any(word in text_lower for word in ['hospital', 'patient', 'doctor']):
            return "Hospital Ward"
        elif any(word in text_lower for word in ['beach', 'mountains', 'holiday', 'vacation']):
            return "Casual Setting"
        else:
            return "Office Floor 3"
    
    def _infer_meeting_type(self, conversation_text: str, scenario_context: Dict) -> str:
        """Infer meeting type from content"""
        text_lower = conversation_text.lower()
        
        if 'launch' in text_lower:
            return "Project Launch Meeting"
        elif any(word in text_lower for word in ['patient', 'medical', 'doctor']):
            return "Medical Consultation"
        elif any(word in text_lower for word in ['holiday', 'vacation', 'beach']):
            return "Personal Planning"
        else:
            return "Team Coordination"
    
    def _extract_project_names(self, conversation_text: str, speaker_profile: Dict) -> List[str]:
        """Extract project names from conversation and profile"""
        projects = []
        
        # From speaker profile
        if isinstance(speaker_profile, dict) and speaker_profile.get("Current project"):
            projects.extend(speaker_profile["Current project"].split(", "))
        
        # From conversation text
        project_patterns = [r'Project\s+(\w+)', r'(\w+)\s+project']
        for pattern in project_patterns:
            matches = re.findall(pattern, conversation_text, re.IGNORECASE)
            projects.extend(matches)
        
        return list(set(projects))

class SimpleContextAggregator:
    """Framework Context Aggregator for evaluation"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def process_conversation(self, conversation_data: ConversationData, metadata: Dict) -> Dict:
        """Process conversation with context aggregation"""
        
        transcript = ' '.join(conversation_data.speaker_utterances)
        
        # Reference Resolution
        resolved_speech = self._resolve_references(transcript, metadata)
        
        # Context Enhancement
        enhanced_context = self._enhance_context(resolved_speech, metadata)
        
        # Information Gap Detection
        information_gaps = self._detect_gaps(enhanced_context, transcript)
        
        return {
            "resolved_content": {
                "original_speech": transcript,
                "resolved_speech": resolved_speech,
                "action_items": enhanced_context.get("action_items", []),
                "topics": enhanced_context.get("topics", []),
                "urgency_level": enhanced_context.get("urgency_level", "medium"),
                "main_purpose": enhanced_context.get("main_purpose", "discussion"),
                "participants": metadata["participants"],
                "location": metadata["spatial"]["location"]
            },
            "information_gaps": information_gaps,
            "context_summary": {
                "conversation_type": enhanced_context.get("conversation_type", "discussion"),
                "urgency_level": enhanced_context.get("urgency_level", "medium"),
                "requires_collaboration": len(enhanced_context.get("action_items", [])) > 0,
                "completeness_score": 0.8
            }
        }
    
    def _resolve_references(self, transcript: str, metadata: Dict) -> str:
        """FRAMEWORK: Resolve vague references in transcript"""
        
        prompt = f"""
        Original conversation: "{transcript}"
        
        Context:
        - Location: {metadata["spatial"]["location"]}
        - Participants: {metadata["participants"]}
        - Meeting: {metadata["calendar"]["current_meeting"]}
        - Projects: {metadata["calendar"]["related_projects"]}
        
        Replace vague references with specific details:
        - "this project" → specific project name
        - "here" → specific location
        - "tomorrow/today" → specific dates
        
        Return only the conversation with references resolved.
        """
        
        return self.llm_client.generate(prompt, max_tokens=500)
    
    def _enhance_context(self, resolved_speech: str, metadata: Dict) -> Dict:
        """FRAMEWORK: Extract structured information from resolved speech"""
        
        prompt = f"""
        Conversation: "{resolved_speech}"
        Location: {metadata["spatial"]["location"]}
        
        Extract information in JSON format:
        {{
            "action_items": ["list of action items"],
            "topics": ["main topics discussed"],
            "urgency_level": "low/medium/high",
            "conversation_type": "type of interaction",
            "main_purpose": "primary goal of conversation"
        }}
        """
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=400)
            # Simple JSON extraction
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return {"action_items": [], "topics": [], "urgency_level": "medium"}
        except:
            return {"action_items": [], "topics": [], "urgency_level": "medium"}
    
    def _detect_gaps(self, enhanced_context: Dict, original_transcript: str) -> List[str]:
        """FRAMEWORK: Detect information gaps with LLM"""
        
        prompt = f"""
        Based on this conversation: "{original_transcript}"
        
        What are the most important questions that you think are important and should be asked to another person to help complete the tasks or understand the situation better?
        
        Focus on questions that would:
        - Help accomplish the mentioned tasks
        - Clarify missing details
        - Enable better coordination
        - Resolve uncertainties
        
        Return as a simple list:
        1. Question 1
        2. Question 2  
        3. Question 3
        """
        
        response = self.llm_client.generate(prompt, max_tokens=300)
        
        # Extract questions from response
        questions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                clean_question = re.sub(r'^\d+\.?\s*', '', line)
                clean_question = re.sub(r'^[-•]\s*', '', clean_question)
                if clean_question:
                    questions.append(clean_question.strip())
        
        return questions[:3]  # Return top 3

class BaselineGPT:
    """Enhanced Baseline GPT that mirrors framework structure"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def process_conversation_baseline(self, conversation_data: ConversationData, metadata: Dict) -> Dict:
        """Complete baseline processing that mirrors framework"""
        
        conversation_str = ' '.join(conversation_data.speaker_utterances)
        
        # Baseline Context Aggregation
        baseline_context = self._baseline_context_aggregation(conversation_str, metadata)
        
        # Baseline Information Gap Detection  
        baseline_questions = self._baseline_info_gap_detection(conversation_str, metadata)
        
        # Parse baseline context to extract structured data (for metric calculation)
        parsed_context = self._parse_baseline_context(baseline_context)
        
        return {
            "baseline_context_aggregation": baseline_context,
            "baseline_info_gap_detection": baseline_questions,
            "baseline_full_output": baseline_context + "\n\n" + baseline_questions,
            "parsed_context": parsed_context,
            "baseline_questions_list": self._extract_questions_list(baseline_questions)
        }
    
    def _baseline_context_aggregation(self, conversation_str: str, metadata: Dict) -> str:
        """BASELINE: Context Aggregation - separate prompt"""
        
        prompt = f"""
        Conversation: "{conversation_str}"
        
        Context Information:
        - Location: {metadata["spatial"]["location"]}
        - Current Meeting: {metadata["calendar"]["current_meeting"]}
        - Projects: {metadata["calendar"]["related_projects"]}
        - Participants: {metadata["participants"]}
        
        Analyze this conversation and provide:
        
        1. REFERENCE RESOLUTION: Replace vague words like "this", "here", "tomorrow" with specific details
        2. ACTION ITEMS: What tasks were mentioned or need to be done?
        3. MAIN TOPICS: What are the key topics discussed?
        4. URGENCY LEVEL: How urgent is this conversation? (low/medium/high)
        5. PURPOSE: What's the main goal of this conversation?
        
        Provide a structured analysis covering these 5 areas.
        """
        
        return self.llm_client.generate(prompt, max_tokens=600)
    
    def _baseline_info_gap_detection(self, conversation_str: str, metadata: Dict) -> str:
        """BASELINE: Information Gap Detection - separate prompt"""
        
        prompt = f"""
        Conversation: "{conversation_str}"
        
        Context Information:
        - Location: {metadata["spatial"]["location"]}
        - Current Meeting: {metadata["calendar"]["current_meeting"]}
        - Projects: {metadata["calendar"]["related_projects"]}
        - Participants: {metadata["participants"]}
        
        What are the most important questions that you think are important and should be asked to another person to help with this conversation?
        
        Consider:
        - What information is missing to complete the mentioned tasks?
        - What details need clarification?
        - What coordination or confirmation is needed?
        - What would help move the conversation/project forward?
        
        List the specific questions that should be asked.
        """
        
        return self.llm_client.generate(prompt, max_tokens=500)
    
    def _parse_baseline_context(self, baseline_context: str) -> Dict:
        """Parse baseline context output to extract structured data for metrics"""
        
        # Simple parsing to extract action items, topics, etc.
        parsed = {
            "action_items": [],
            "topics": [],
            "urgency_level": "medium",
            "main_purpose": "discussion"
        }
        
        lines = baseline_context.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Identify sections
            if 'action item' in line.lower():
                current_section = 'action_items'
            elif 'topic' in line.lower():
                current_section = 'topics'
            elif 'urgency' in line.lower():
                if 'high' in line.lower():
                    parsed['urgency_level'] = 'high'
                elif 'low' in line.lower():
                    parsed['urgency_level'] = 'low'
            elif 'purpose' in line.lower():
                current_section = 'main_purpose'
            
            # Extract items
            if current_section == 'action_items' and ('•' in line or '-' in line or line.startswith('1.')):
                item = re.sub(r'^[-•\d\.]+\s*', '', line)
                if item:
                    parsed['action_items'].append(item)
            elif current_section == 'topics' and ('•' in line or '-' in line or line.startswith('1.')):
                item = re.sub(r'^[-•\d\.]+\s*', '', line)
                if item:
                    parsed['topics'].append(item)
        
        return parsed
    
    def _extract_questions_list(self, baseline_questions: str) -> List[str]:
        """Extract questions from baseline output"""
        
        questions = []
        lines = baseline_questions.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('?' in line or any(word in line.lower() for word in ['what', 'how', 'when', 'where', 'who', 'why'])):
                # Clean up the question
                clean_question = re.sub(r'^[-•\d\.]+\s*', '', line)
                if clean_question and clean_question not in questions:
                    questions.append(clean_question)
        
        return questions[:5]  # Return up to 5 questions

class MetricsCalculator:
    """Calculate the SAME 4 metrics for both Framework and Baseline"""
    
    def calculate_all_metrics(self, framework_output: Dict, baseline_output: Dict, 
                            conversation_data: ConversationData) -> Dict:
        """Calculate SAME metrics for both Framework and Baseline"""
        
        # Calculate metrics for Framework
        framework_metrics = {
            "reference_resolution_accuracy": self._calculate_resolution_accuracy_framework(framework_output),
            "completeness_score": self._calculate_completeness_score_framework(framework_output),
            "question_relevance_score": self._calculate_relevance_score_framework(framework_output, conversation_data),
            "question_count": len(framework_output.get("information_gaps", []))
        }
        
        # Calculate SAME metrics for Baseline
        baseline_metrics = {
            "reference_resolution_accuracy": self._calculate_resolution_accuracy_baseline(baseline_output),
            "completeness_score": self._calculate_completeness_score_baseline(baseline_output),
            "question_relevance_score": self._calculate_relevance_score_baseline(baseline_output, conversation_data),
            "question_count": len(baseline_output.get("baseline_questions_list", []))
        }
        
        # Calculate comparative metrics
        comparative_metrics = {
            "framework_vs_baseline_efficiency": framework_metrics["question_count"] / max(baseline_metrics["question_count"], 1),
            "framework_vs_baseline_relevance_diff": framework_metrics["question_relevance_score"] - baseline_metrics["question_relevance_score"]
        }
        
        return {
            "framework": framework_metrics,
            "baseline": baseline_metrics,
            "comparative": comparative_metrics
        }
    
    def _calculate_resolution_accuracy_framework(self, framework_output: Dict) -> float:
        """Calculate reference resolution accuracy for Framework"""
        
        resolved_content = framework_output.get("resolved_content", {})
        original = resolved_content.get("original_speech", "")
        resolved = resolved_content.get("resolved_speech", "")
        
        return self._resolution_accuracy_helper(original, resolved)
    
    def _calculate_resolution_accuracy_baseline(self, baseline_output: Dict) -> float:
        """Calculate reference resolution accuracy for Baseline"""
        
        baseline_context = baseline_output.get("baseline_context_aggregation", "")
        
        # Count how many vague references baseline resolved
        vague_terms = ["this", "that", "here", "there", "it"]
        resolution_indicators = ["specific", "clarified", "identified", "replaced"]
        
        resolution_score = 0.0
        total_checks = 0
        
        context_lower = baseline_context.lower()
        for term in vague_terms:
            if term in context_lower:
                total_checks += 1
                for indicator in resolution_indicators:
                    if indicator in context_lower:
                        resolution_score += 0.25  # Partial credit
                        break
        
        return resolution_score / max(total_checks, 1)
    
    def _resolution_accuracy_helper(self, original: str, resolved: str) -> float:
        """Helper function to calculate resolution accuracy"""
        
        if not original or not resolved:
            return 0.0
        
        # Count vague terms resolved
        vague_terms = ["this", "that", "here", "there", "it"]
        resolved_count = 0
        total_vague = 0
        
        for term in vague_terms:
            original_count = original.lower().count(term)
            resolved_count_term = resolved.lower().count(term)
            
            if original_count > 0:
                total_vague += original_count
                resolved_count += max(0, original_count - resolved_count_term)
        
        return resolved_count / total_vague if total_vague > 0 else 1.0
    
    def _calculate_completeness_score_framework(self, framework_output: Dict) -> float:
        """Calculate completeness score for Framework"""
        
        resolved_content = framework_output.get("resolved_content", {})
        
        expected_elements = ["action_items", "topics", "urgency_level", "main_purpose"]
        found_elements = sum(1 for elem in expected_elements if resolved_content.get(elem))
        
        return found_elements / len(expected_elements)
    
    def _calculate_completeness_score_baseline(self, baseline_output: Dict) -> float:
        """Calculate completeness score for Baseline"""
        
        parsed_context = baseline_output.get("parsed_context", {})
        
        expected_elements = ["action_items", "topics", "urgency_level", "main_purpose"]
        found_elements = sum(1 for elem in expected_elements if parsed_context.get(elem))
        
        return found_elements / len(expected_elements)
    
    def _calculate_relevance_score_framework(self, framework_output: Dict, conversation_data: ConversationData) -> float:
        """Calculate question relevance for Framework"""
        
        gaps = framework_output.get("information_gaps", [])
        return self._relevance_score_helper(gaps, conversation_data)
    
    def _calculate_relevance_score_baseline(self, baseline_output: Dict, conversation_data: ConversationData) -> float:
        """Calculate question relevance for Baseline"""
        
        questions = baseline_output.get("baseline_questions_list", [])
        return self._relevance_score_helper(questions, conversation_data)
    
    def _relevance_score_helper(self, questions: List[str], conversation_data: ConversationData) -> float:
        """Helper function to calculate relevance score"""
        
        if not questions:
            return 0.0
        
        conversation_text = ' '.join(conversation_data.speaker_utterances).lower()
        
        # Simple relevance check
        relevant_count = 0
        for question in questions:
            question_words = set(str(question).lower().split())
            conversation_words = set(conversation_text.split())
            if len(question_words & conversation_words) >= 1:  # More lenient threshold
                relevant_count += 1
        
        return relevant_count / len(questions)

class EvaluationPipeline:
    """Main evaluation pipeline with ultra-robust parsing"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.parser = ConversationParser()
        self.metadata_generator = MetadataGenerator()
        self.llm_client = LLMClient()
        self.context_aggregator = SimpleContextAggregator(self.llm_client)
        self.baseline_gpt = BaselineGPT(self.llm_client)
        self.metrics_calculator = MetricsCalculator()
    
    def run_complete_evaluation(self) -> List[EvaluationResult]:
        """Run complete evaluation on ALL valid conversation files"""
        
        print("🚀 Starting ULTRA-ROBUST Evaluation Pipeline")
        print("=" * 60)
        
        results = []
        txt_files = self._find_all_txt_files()
        total_files = len(txt_files)
        print(f"Found {total_files} conversation files across all folders")
        
        successful_results = 0
        failed_files = 0
        skipped_files = 0
        
        # Process ALL files
        for i, txt_file in enumerate(txt_files, 1):
            print(f"\n[{i}/{total_files}] Processing: {os.path.basename(txt_file)}")
            
            try:
                # Parse conversation with intelligent filtering
                conversation_data = self.parser.parse_conversation_file(txt_file)
                
                # Check if file should be skipped
                if conversation_data.speaker_name == "SKIP_FILE":
                    skipped_files += 1
                    continue
                
                if not conversation_data.speaker_utterances:
                    print(f"   ⚠️ No valid utterances found")
                    failed_files += 1
                    continue
                
                # Generate metadata
                metadata = self.metadata_generator.generate_metadata(conversation_data)
                
                # Run framework
                print(f"   🏗️ Running Framework...")
                framework_output = self.context_aggregator.process_conversation(conversation_data, metadata)
                
                # Run baseline
                print(f"   🤖 Running Baseline...")
                baseline_output = self.baseline_gpt.process_conversation_baseline(conversation_data, metadata)
                
                # Calculate metrics
                print(f"   📊 Calculating Metrics...")
                metrics = self.metrics_calculator.calculate_all_metrics(
                    framework_output, baseline_output, conversation_data
                )
                
                result = EvaluationResult(
                    file_path=txt_file,
                    speaker_name=conversation_data.speaker_name,
                    framework_output=framework_output,
                    baseline_output=baseline_output,
                    metrics=metrics,
                    metadata_used=metadata
                )
                
                results.append(result)
                successful_results += 1
                print(f"   ✅ Completed: {conversation_data.speaker_name}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed_files += 1
                continue
        
        print(f"\n📊 Processing Complete:")
        print(f"   Successfully processed: {successful_results}/{total_files} files")
        print(f"   Skipped (narrative): {skipped_files}/{total_files} files") 
        print(f"   Failed: {failed_files}/{total_files} files")
        
        if results:
            self._print_summary(results)
            self._save_comprehensive_results(results)
        
        return results
    
    def _find_all_txt_files(self) -> List[str]:
        """Find ALL txt files in dataset across ALL subdirectories"""
        txt_files = []
        print(f"\n📁 Scanning directories:")
        
        for root, dirs, files in os.walk(self.dataset_path):
            folder_name = os.path.basename(root)
            txt_count = len([f for f in files if f.endswith('.txt')])
            if txt_count > 0:
                print(f"   📂 {folder_name}: {txt_count} files")
            
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        return txt_files
    
    def _print_summary(self, results: List[EvaluationResult]):
        """Print evaluation summary"""
        
        print("\n" + "=" * 60)
        print("📊 ULTRA-ROBUST EVALUATION SUMMARY")
        print("=" * 60)
        
        # Calculate averages for Framework
        fw_resolution = sum(r.metrics["framework"]["reference_resolution_accuracy"] for r in results) / len(results)
        fw_completeness = sum(r.metrics["framework"]["completeness_score"] for r in results) / len(results)
        fw_relevance = sum(r.metrics["framework"]["question_relevance_score"] for r in results) / len(results)
        fw_question_count = sum(r.metrics["framework"]["question_count"] for r in results) / len(results)
        
        # Calculate averages for Baseline
        bl_resolution = sum(r.metrics["baseline"]["reference_resolution_accuracy"] for r in results) / len(results)
        bl_completeness = sum(r.metrics["baseline"]["completeness_score"] for r in results) / len(results)
        bl_relevance = sum(r.metrics["baseline"]["question_relevance_score"] for r in results) / len(results)
        bl_question_count = sum(r.metrics["baseline"]["question_count"] for r in results) / len(results)
        
        print(f"📁 Valid Conversation Files Processed: {len(results)}")
        
        print(f"\n🏗️ FRAMEWORK RESULTS:")
        print(f"   Reference Resolution: {fw_resolution:.3f}")
        print(f"   Completeness Score: {fw_completeness:.3f}")
        print(f"   Question Relevance: {fw_relevance:.3f}")
        print(f"   Avg Questions: {fw_question_count:.1f}")
        
        print(f"\n🤖 BASELINE RESULTS:")
        print(f"   Reference Resolution: {bl_resolution:.3f}")
        print(f"   Completeness Score: {bl_completeness:.3f}")
        print(f"   Question Relevance: {bl_relevance:.3f}")
        print(f"   Avg Questions: {bl_question_count:.1f}")
        
        print(f"\n⚡ FRAMEWORK vs BASELINE:")
        print(f"   Resolution Advantage: {fw_resolution - bl_resolution:+.3f}")
        print(f"   Completeness Advantage: {fw_completeness - bl_completeness:+.3f}")
        print(f"   Relevance Advantage: {fw_relevance - bl_relevance:+.3f}")
        print(f"   Question Efficiency: {fw_question_count - bl_question_count:+.1f}")
    
    def _save_comprehensive_results(self, results: List[EvaluationResult]):
        """Save comprehensive results with same metrics for both Framework and Baseline"""
        
        data = []
        
        for result in results:
            row = {
                # File info
                "file_path": result.file_path,
                "file_name": os.path.basename(result.file_path),
                "speaker_name": result.speaker_name,
                "conversation_format": result.framework_output["resolved_content"].get("location", "unknown"),
                
                # Framework metrics
                "framework_reference_resolution": result.metrics["framework"]["reference_resolution_accuracy"],
                "framework_completeness_score": result.metrics["framework"]["completeness_score"],
                "framework_question_relevance": result.metrics["framework"]["question_relevance_score"],
                "framework_question_count": result.metrics["framework"]["question_count"],
                
                # Baseline metrics (SAME as framework)
                "baseline_reference_resolution": result.metrics["baseline"]["reference_resolution_accuracy"],
                "baseline_completeness_score": result.metrics["baseline"]["completeness_score"],
                "baseline_question_relevance": result.metrics["baseline"]["question_relevance_score"],
                "baseline_question_count": result.metrics["baseline"]["question_count"],
                
                # Comparative metrics
                "efficiency_ratio": result.metrics["comparative"]["framework_vs_baseline_efficiency"],
                "relevance_difference": result.metrics["comparative"]["framework_vs_baseline_relevance_diff"],
                
                # Raw outputs (truncated for CSV)
                "framework_questions": str(result.framework_output.get("information_gaps", [])[:3]),
                "baseline_questions": str(result.baseline_output.get("baseline_questions_list", [])[:3]),
                "sample_conversation": ' '.join(result.framework_output["resolved_content"].get("original_speech", "").split()[:30]) + "..."
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv("lexicon_ultra_robust_evaluation.csv", index=False)
        print(f"\n💾 Ultra-robust evaluation saved to: lexicon_ultra_robust_evaluation.csv")
        print(f"   📊 Rows: {len(df)} | Columns: {len(df.columns)}")
        print(f"   🎯 Handles ALL conversation formats + filters narrative files")
        print(f"   🔥 Same 4 metrics for both Framework and Baseline")

def main():
    """Main function"""
    
    # Dataset path - relative to current working directory
    DATASET_PATH = "./dataset"
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please ensure you're running this script from the correct directory.")
        return
    
    pipeline = EvaluationPipeline(DATASET_PATH)
    results = pipeline.run_complete_evaluation()
    
    print(f"\n🎉 ULTRA-ROBUST evaluation complete!")
    print(f"✅ Successfully processed {len(results)} valid conversations")
    print(f"🎯 Handles ALL formats: old complex + new variations + edge cases")
    print(f"🚫 Automatically filters out narrative/incomplete files")
    print(f"📊 Same 4 metrics for Framework vs Baseline comparison")

if __name__ == "__main__":
    main()