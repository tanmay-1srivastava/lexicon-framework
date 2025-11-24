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
class GroundTruthAnnotation:
    timestamp: str
    speaker: str
    word_or_phrase: str
    refers_to: str
    importance: int

@dataclass
class EvaluationResult:
    file_path: str
    speaker_name: str
    target_user_utterances: List[str]
    ground_truth_annotations: List[GroundTruthAnnotation]
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
    """Parser that handles conversation formats with ground truth annotations"""
    
    def parse_conversation_file(self, file_path: str) -> Tuple[ConversationData, List[GroundTruthAnnotation]]:
        """Parse conversation file and extract ground truth annotations"""
        
        print(f"📂 Processing: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Step 1: Check if file contains actual conversation
        if not self._contains_actual_conversation(content):
            print("   ⚠️ SKIPPED: File contains narrative/description, not actual dialogue")
            return self._create_empty_conversation_data(), []
        
        # Step 2: Extract annotations section
        annotations = self._extract_annotations(content)
        print(f"   📝 Found {len(annotations)} ground truth annotations")
        
        # Step 3: Parse conversation (focus on new format with timestamps)
        conversation_data = self._parse_timestamped_format(content, file_path)
        
        return conversation_data, annotations
    
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
    
    def _extract_annotations(self, content: str) -> List[GroundTruthAnnotation]:
        """Extract ground truth annotations from the annotations section"""
        annotations = []
        
        try:
            # Find annotations section
            annotations_start = content.find('// ---Annotations Section---')
            if annotations_start == -1:
                print("   ⚠️ No annotations section found")
                return annotations
            
            annotations_text = content[annotations_start:]
            
            # Extract JSON array from annotations
            start_bracket = annotations_text.find('[')
            if start_bracket == -1:
                return annotations
                
            # Find the matching closing bracket
            bracket_count = 0
            end_bracket = -1
            for i, char in enumerate(annotations_text[start_bracket:], start_bracket):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_bracket = i + 1
                        break
            
            if end_bracket != -1:
                json_text = annotations_text[start_bracket:end_bracket]
                annotations_data = json.loads(json_text)
                
                for annotation in annotations_data:
                    if all(key in annotation for key in ['timestamp', 'speaker', 'notes']):
                        notes = annotation['notes']
                        if all(key in notes for key in ['word_or_phrase', 'refers_to', 'importance']):
                            annotations.append(GroundTruthAnnotation(
                                timestamp=annotation['timestamp'],
                                speaker=annotation['speaker'],
                                word_or_phrase=notes['word_or_phrase'],
                                refers_to=notes['refers_to'],
                                importance=notes['importance']
                            ))
        
        except Exception as e:
            print(f"   ⚠️ Error parsing annotations: {e}")
        
        return annotations
    
    def _parse_timestamped_format(self, content: str, file_path: str) -> ConversationData:
        """Parse the timestamped conversation format"""
        
        print("   Detected: Timestamped conversation format")
        
        # Extract conversation part (before annotations)
        conversation_end = content.find('// ---Annotations Section---')
        if conversation_end != -1:
            conversation_content = content[:conversation_end]
        else:
            conversation_content = content
        
        # Extract timestamped utterances
        pattern = r'(\d{2}:\d{2}:\d{2}):\s*\{\s*([A-Za-z][A-Za-z0-9_\s]*?):\s*"([^"]+)"\s*\}'
        matches = re.findall(pattern, conversation_content, re.DOTALL)
        
        if not matches:
            print("   ⚠️ No timestamped utterances found")
            return self._create_empty_conversation_data()
        
        # Get all speakers and their utterances
        all_speakers = list(set(match[1].strip() for match in matches))
        print(f"   Found speakers: {all_speakers}")
        
        # Choose target user (first speaker with most utterances)
        speaker_counts = {}
        for _, speaker, _ in matches:
            speaker = speaker.strip()
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        target_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
        print(f"   Target user selected: {target_speaker} ({speaker_counts[target_speaker]} utterances)")
        
        # Extract target user's utterances and timestamps
        target_utterances = []
        target_timestamps = []
        
        for timestamp, speaker, utterance in matches:
            speaker = speaker.strip()
            if speaker == target_speaker:
                target_timestamps.append(timestamp)
                clean_utterance = utterance.strip()
                target_utterances.append(clean_utterance)
        
        print(f"   Extracted {len(target_utterances)} utterances for target user")
        
        # Create speaker profiles
        other_speakers = [s for s in all_speakers if s != target_speaker]
        speaker_profile = {"Name": target_speaker, "Role": "Target User"}
        other_speaker_profile = {"Name": other_speakers[0] if other_speakers else "Other", "Role": "Participant"}
        
        return ConversationData(
            speaker_name=target_speaker,
            speaker_utterances=target_utterances,
            timestamps=target_timestamps,
            speaker_profile=speaker_profile,
            other_speaker_profile=other_speaker_profile,
            scenario_context={"format": "timestamped", "file": os.path.basename(file_path)}
        )
    
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
    """Framework Context Aggregator for evaluation - focuses on target user"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def process_conversation(self, conversation_data: ConversationData, metadata: Dict, 
                           ground_truth_annotations: List[GroundTruthAnnotation]) -> Dict:
        """Process target user's conversation part with context aggregation"""
        
        # Focus only on target user's utterances
        target_transcript = ' '.join(conversation_data.speaker_utterances)
        target_speaker = conversation_data.speaker_name
        
        print(f"   🎯 Processing {target_speaker}'s utterances: {len(conversation_data.speaker_utterances)} total")
        
        # Reference Resolution for target user's speech
        resolved_speech = self._resolve_references(target_transcript, metadata, target_speaker)
        
        # Context Enhancement
        enhanced_context = self._enhance_context(resolved_speech, metadata)
        
        # Information Gap Detection based on target user's needs
        information_gaps = self._detect_gaps(enhanced_context, target_transcript, target_speaker)
        
        return {
            "resolved_content": {
                "original_speech": target_transcript,
                "resolved_speech": resolved_speech,
                "target_speaker": target_speaker,
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
    
    def _resolve_references(self, transcript: str, metadata: Dict, target_speaker: str) -> str:
        """FRAMEWORK: Resolve vague references in target user's speech"""
        
        prompt = f"""
        Target User: {target_speaker}
        Target User's Speech: "{transcript}"
        
        Context:
        - Location: {metadata["spatial"]["location"]}
        - Participants: {metadata["participants"]}
        - Meeting: {metadata["calendar"]["current_meeting"]}
        - Projects: {metadata["calendar"]["related_projects"]}
        
        Resolve vague references in {target_speaker}'s speech with specific details:
        - "this project" → specific project name
        - "here/there" → specific location
        - "tomorrow/today" → specific dates  
        - "it/that/this" → specific objects
        - "then/now" → specific times
        
        Return only {target_speaker}'s speech with references resolved.
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
    
    def _detect_gaps(self, enhanced_context: Dict, original_transcript: str, target_speaker: str) -> List[str]:
        """FRAMEWORK: Detect information gaps from target user's perspective"""
        
        prompt = f"""
        Target User: {target_speaker}
        Target User's Speech: "{original_transcript}"
        
        Based on what {target_speaker} said, what are the most important and useful questions that {target_speaker} should ask to others to:
        - Complete their mentioned tasks
        - Get clarification on unclear points
        - Coordinate better with others
        - Resolve uncertainties they have
        - Make informed decisions
        
        Focus on practical, actionable questions that would genuinely help {target_speaker}.
        
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
    """Enhanced Baseline GPT that focuses on target user"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def process_conversation_baseline(self, conversation_data: ConversationData, metadata: Dict) -> Dict:
        """Complete baseline processing focusing on target user"""
        
        conversation_str = ' '.join(conversation_data.speaker_utterances)
        target_speaker = conversation_data.speaker_name
        
        # Baseline Context Aggregation
        baseline_context = self._baseline_context_aggregation(conversation_str, metadata, target_speaker)
        
        # Baseline Information Gap Detection  
        baseline_questions = self._baseline_info_gap_detection(conversation_str, metadata, target_speaker)
        
        # Parse baseline context to extract structured data (for metric calculation)
        parsed_context = self._parse_baseline_context(baseline_context)
        
        return {
            "baseline_context_aggregation": baseline_context,
            "baseline_info_gap_detection": baseline_questions,
            "baseline_full_output": baseline_context + "\n\n" + baseline_questions,
            "parsed_context": parsed_context,
            "baseline_questions_list": self._extract_questions_list(baseline_questions)
        }
    
    def _baseline_context_aggregation(self, conversation_str: str, metadata: Dict, target_speaker: str) -> str:
        """BASELINE: Context Aggregation focusing on target user"""
        
        prompt = f"""
        Target User: {target_speaker}
        Target User's Speech: "{conversation_str}"
        
        Context Information:
        - Location: {metadata["spatial"]["location"]}
        - Current Meeting: {metadata["calendar"]["current_meeting"]}
        - Projects: {metadata["calendar"]["related_projects"]}
        - Participants: {metadata["participants"]}
        
        Analyze {target_speaker}'s speech and provide:
        
        1. REFERENCE RESOLUTION: Replace vague words like "this", "here", "tomorrow" with specific details
        2. ACTION ITEMS: What tasks did {target_speaker} mention or need to do?
        3. MAIN TOPICS: What key topics did {target_speaker} discuss?
        4. URGENCY LEVEL: How urgent are {target_speaker}'s concerns? (low/medium/high)
        5. PURPOSE: What's {target_speaker}'s main goal?
        
        Provide a structured analysis covering these 5 areas.
        """
        
        return self.llm_client.generate(prompt, max_tokens=600)
    
    def _baseline_info_gap_detection(self, conversation_str: str, metadata: Dict, target_speaker: str) -> str:
        """BASELINE: Information Gap Detection focusing on target user"""
        
        prompt = f"""
        Target User: {target_speaker}
        Target User's Speech: "{conversation_str}"
        
        Context Information:
        - Location: {metadata["spatial"]["location"]}
        - Current Meeting: {metadata["calendar"]["current_meeting"]}
        - Projects: {metadata["calendar"]["related_projects"]}
        - Participants: {metadata["participants"]}
        
        What are the most important questions that {target_speaker} should ask to others based on what they said?
        
        Consider:
        - What information does {target_speaker} need to complete their tasks?
        - What details does {target_speaker} need clarified?
        - What coordination does {target_speaker} need?
        - What would help {target_speaker} move forward?
        
        List the specific questions {target_speaker} should ask.
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
    """Calculate metrics using ground truth annotations"""
    
    def calculate_all_metrics(self, framework_output: Dict, baseline_output: Dict, 
                            conversation_data: ConversationData, 
                            ground_truth_annotations: List[GroundTruthAnnotation]) -> Dict:
        """Calculate comprehensive metrics using ground truth"""
        
        # Calculate ground truth based metrics for Framework
        framework_metrics = {
            "reference_resolution_precision": self._calculate_reference_precision_framework(
                framework_output, ground_truth_annotations, conversation_data),
            "reference_resolution_recall": self._calculate_reference_recall_framework(
                framework_output, ground_truth_annotations, conversation_data),
            "temporal_resolution_accuracy": self._calculate_temporal_accuracy_framework(
                framework_output, ground_truth_annotations),
            "spatial_resolution_accuracy": self._calculate_spatial_accuracy_framework(
                framework_output, ground_truth_annotations),
            "object_resolution_accuracy": self._calculate_object_accuracy_framework(
                framework_output, ground_truth_annotations),
            "question_usefulness_score": self._calculate_question_usefulness_framework(
                framework_output, conversation_data),
            "question_count": len(framework_output.get("information_gaps", []))
        }
        
        # Calculate same metrics for Baseline
        baseline_metrics = {
            "reference_resolution_precision": self._calculate_reference_precision_baseline(
                baseline_output, ground_truth_annotations, conversation_data),
            "reference_resolution_recall": self._calculate_reference_recall_baseline(
                baseline_output, ground_truth_annotations, conversation_data),
            "temporal_resolution_accuracy": self._calculate_temporal_accuracy_baseline(
                baseline_output, ground_truth_annotations),
            "spatial_resolution_accuracy": self._calculate_spatial_accuracy_baseline(
                baseline_output, ground_truth_annotations),
            "object_resolution_accuracy": self._calculate_object_accuracy_baseline(
                baseline_output, ground_truth_annotations),
            "question_usefulness_score": self._calculate_question_usefulness_baseline(
                baseline_output, conversation_data),
            "question_count": len(baseline_output.get("baseline_questions_list", []))
        }
        
        # Calculate overall scores
        framework_overall = (framework_metrics["reference_resolution_precision"] + 
                           framework_metrics["reference_resolution_recall"] +
                           framework_metrics["temporal_resolution_accuracy"] +
                           framework_metrics["spatial_resolution_accuracy"] +
                           framework_metrics["object_resolution_accuracy"] +
                           framework_metrics["question_usefulness_score"]) / 6
        
        baseline_overall = (baseline_metrics["reference_resolution_precision"] + 
                          baseline_metrics["reference_resolution_recall"] +
                          baseline_metrics["temporal_resolution_accuracy"] +
                          baseline_metrics["spatial_resolution_accuracy"] +
                          baseline_metrics["object_resolution_accuracy"] +
                          baseline_metrics["question_usefulness_score"]) / 6
        
        # Comparative metrics
        comparative_metrics = {
            "overall_framework_advantage": framework_overall - baseline_overall,
            "reference_resolution_advantage": (framework_metrics["reference_resolution_precision"] + 
                                              framework_metrics["reference_resolution_recall"]) / 2 - 
                                             (baseline_metrics["reference_resolution_precision"] + 
                                              baseline_metrics["reference_resolution_recall"]) / 2,
            "context_aggregation_advantage": (framework_metrics["temporal_resolution_accuracy"] +
                                             framework_metrics["spatial_resolution_accuracy"] +
                                             framework_metrics["object_resolution_accuracy"]) / 3 - 
                                            (baseline_metrics["temporal_resolution_accuracy"] +
                                             baseline_metrics["spatial_resolution_accuracy"] +
                                             baseline_metrics["object_resolution_accuracy"]) / 3,
            "question_quality_advantage": framework_metrics["question_usefulness_score"] - 
                                         baseline_metrics["question_usefulness_score"]
        }
        
        return {
            "framework": framework_metrics,
            "baseline": baseline_metrics,
            "comparative": comparative_metrics
        }
    
    def _calculate_reference_precision_framework(self, framework_output: Dict, 
                                               ground_truth_annotations: List[GroundTruthAnnotation],
                                               conversation_data: ConversationData) -> float:
        """Calculate how many resolved references are correct (Precision)"""
        
        resolved_speech = framework_output.get("resolved_content", {}).get("resolved_speech", "")
        original_speech = framework_output.get("resolved_content", {}).get("original_speech", "")
        
        return self._reference_precision_helper(resolved_speech, original_speech, 
                                              ground_truth_annotations, conversation_data)
    
    def _calculate_reference_precision_baseline(self, baseline_output: Dict, 
                                              ground_truth_annotations: List[GroundTruthAnnotation],
                                              conversation_data: ConversationData) -> float:
        """Calculate reference precision for baseline"""
        
        baseline_context = baseline_output.get("baseline_context_aggregation", "")
        original_speech = ' '.join(conversation_data.speaker_utterances)
        
        # Extract resolved references from baseline context
        return self._reference_precision_helper(baseline_context, original_speech, 
                                              ground_truth_annotations, conversation_data)
    
    def _reference_precision_helper(self, resolved_text: str, original_text: str,
                                   ground_truth_annotations: List[GroundTruthAnnotation],
                                   conversation_data: ConversationData) -> float:
        """Helper to calculate reference precision"""
        
        if not resolved_text or not ground_truth_annotations:
            return 0.0
        
        target_speaker = conversation_data.speaker_name
        correct_resolutions = 0
        attempted_resolutions = 0
        
        # Check each ground truth annotation for target speaker
        for annotation in ground_truth_annotations:
            if annotation.speaker != target_speaker:
                continue
                
            vague_phrase = annotation.word_or_phrase.lower()
            correct_resolution = annotation.refers_to.lower()
            
            # Check if original text contained this vague reference
            if vague_phrase in original_text.lower():
                attempted_resolutions += 1
                
                # Check if resolution appears in resolved text
                resolved_lower = resolved_text.lower()
                if any(word in resolved_lower for word in correct_resolution.split()[:3]):
                    correct_resolutions += 1
        
        return correct_resolutions / max(attempted_resolutions, 1)
    
    def _calculate_reference_recall_framework(self, framework_output: Dict, 
                                            ground_truth_annotations: List[GroundTruthAnnotation],
                                            conversation_data: ConversationData) -> float:
        """Calculate how many ground truth references were resolved (Recall)"""
        
        resolved_speech = framework_output.get("resolved_content", {}).get("resolved_speech", "")
        original_speech = framework_output.get("resolved_content", {}).get("original_speech", "")
        
        return self._reference_recall_helper(resolved_speech, original_speech, 
                                           ground_truth_annotations, conversation_data)
    
    def _calculate_reference_recall_baseline(self, baseline_output: Dict, 
                                           ground_truth_annotations: List[GroundTruthAnnotation],
                                           conversation_data: ConversationData) -> float:
        """Calculate reference recall for baseline"""
        
        baseline_context = baseline_output.get("baseline_context_aggregation", "")
        original_speech = ' '.join(conversation_data.speaker_utterances)
        
        return self._reference_recall_helper(baseline_context, original_speech, 
                                           ground_truth_annotations, conversation_data)
    
    def _reference_recall_helper(self, resolved_text: str, original_text: str,
                                ground_truth_annotations: List[GroundTruthAnnotation],
                                conversation_data: ConversationData) -> float:
        """Helper to calculate reference recall"""
        
        if not ground_truth_annotations:
            return 1.0
        
        target_speaker = conversation_data.speaker_name
        total_references = 0
        resolved_references = 0
        
        # Count all ground truth references for target speaker
        for annotation in ground_truth_annotations:
            if annotation.speaker != target_speaker:
                continue
                
            vague_phrase = annotation.word_or_phrase.lower()
            correct_resolution = annotation.refers_to.lower()
            
            # Check if original text contained this vague reference
            if vague_phrase in original_text.lower():
                total_references += 1
                
                # Check if this reference was resolved
                resolved_lower = resolved_text.lower()
                if vague_phrase not in resolved_lower or any(word in resolved_lower for word in correct_resolution.split()[:2]):
                    resolved_references += 1
        
        return resolved_references / max(total_references, 1)
    
    def _calculate_temporal_accuracy_framework(self, framework_output: Dict, 
                                             ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Calculate temporal reference resolution accuracy"""
        resolved_speech = framework_output.get("resolved_content", {}).get("resolved_speech", "")
        return self._temporal_accuracy_helper(resolved_speech, ground_truth_annotations)
    
    def _calculate_temporal_accuracy_baseline(self, baseline_output: Dict, 
                                            ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Calculate temporal accuracy for baseline"""
        baseline_context = baseline_output.get("baseline_context_aggregation", "")
        return self._temporal_accuracy_helper(baseline_context, ground_truth_annotations)
    
    def _temporal_accuracy_helper(self, resolved_text: str, ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Helper to calculate temporal resolution accuracy"""
        temporal_words = ['now', 'then', 'today', 'tomorrow', 'yesterday', 'next week', 'later']
        
        correct = 0
        total = 0
        
        for annotation in ground_truth_annotations:
            if any(temp_word in annotation.word_or_phrase.lower() for temp_word in temporal_words):
                total += 1
                # Check if correct resolution appears
                if any(word in resolved_text.lower() for word in annotation.refers_to.lower().split()[:3]):
                    correct += 1
        
        return correct / max(total, 1)
    
    def _calculate_spatial_accuracy_framework(self, framework_output: Dict, 
                                            ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Calculate spatial reference resolution accuracy"""
        resolved_speech = framework_output.get("resolved_content", {}).get("resolved_speech", "")
        return self._spatial_accuracy_helper(resolved_speech, ground_truth_annotations)
    
    def _calculate_spatial_accuracy_baseline(self, baseline_output: Dict, 
                                           ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Calculate spatial accuracy for baseline"""
        baseline_context = baseline_output.get("baseline_context_aggregation", "")
        return self._spatial_accuracy_helper(baseline_context, ground_truth_annotations)
    
    def _spatial_accuracy_helper(self, resolved_text: str, ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Helper to calculate spatial resolution accuracy"""
        spatial_words = ['here', 'there', 'everywhere', 'somewhere', 'location', 'place']
        
        correct = 0
        total = 0
        
        for annotation in ground_truth_annotations:
            if any(spatial_word in annotation.word_or_phrase.lower() for spatial_word in spatial_words):
                total += 1
                # Check if correct resolution appears
                if any(word in resolved_text.lower() for word in annotation.refers_to.lower().split()[:3]):
                    correct += 1
        
        return correct / max(total, 1)
    
    def _calculate_object_accuracy_framework(self, framework_output: Dict, 
                                           ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Calculate object reference resolution accuracy"""
        resolved_speech = framework_output.get("resolved_content", {}).get("resolved_speech", "")
        return self._object_accuracy_helper(resolved_speech, ground_truth_annotations)
    
    def _calculate_object_accuracy_baseline(self, baseline_output: Dict, 
                                          ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Calculate object accuracy for baseline"""
        baseline_context = baseline_output.get("baseline_context_aggregation", "")
        return self._object_accuracy_helper(baseline_context, ground_truth_annotations)
    
    def _object_accuracy_helper(self, resolved_text: str, ground_truth_annotations: List[GroundTruthAnnotation]) -> float:
        """Helper to calculate object resolution accuracy"""
        object_words = ['this', 'that', 'it', 'they', 'all', 'everything']
        
        correct = 0
        total = 0
        
        for annotation in ground_truth_annotations:
            if any(obj_word in annotation.word_or_phrase.lower() for obj_word in object_words):
                total += 1
                # Check if correct resolution appears
                if any(word in resolved_text.lower() for word in annotation.refers_to.lower().split()[:3]):
                    correct += 1
        
        return correct / max(total, 1)
    
    def _calculate_question_usefulness_framework(self, framework_output: Dict, conversation_data: ConversationData) -> float:
        """Calculate usefulness of framework's information gap questions"""
        gaps = framework_output.get("information_gaps", [])
        return self._question_usefulness_helper(gaps, conversation_data)
    
    def _calculate_question_usefulness_baseline(self, baseline_output: Dict, conversation_data: ConversationData) -> float:
        """Calculate usefulness of baseline's information gap questions"""
        questions = baseline_output.get("baseline_questions_list", [])
        return self._question_usefulness_helper(questions, conversation_data)
    
    def _question_usefulness_helper(self, questions: List[str], conversation_data: ConversationData) -> float:
        """Helper to calculate question usefulness"""
        if not questions:
            return 0.0
        
        conversation_text = ' '.join(conversation_data.speaker_utterances).lower()
        useful_count = 0
        
        # Criteria for useful questions
        useful_indicators = [
            'what', 'when', 'where', 'how', 'who', 'why',  # Question words
            'clarif', 'confirm', 'detail', 'specific',     # Clarification
            'next', 'deadline', 'timeline', 'schedule',    # Planning
            'help', 'support', 'coordinate', 'collaborate' # Collaboration
        ]
        
        for question in questions:
            question_lower = str(question).lower()
            usefulness_score = 0
            
            # Check for useful indicators
            for indicator in useful_indicators:
                if indicator in question_lower:
                    usefulness_score += 1
            
            # Check relevance to conversation
            question_words = set(question_lower.split())
            conversation_words = set(conversation_text.split())
            common_words = len(question_words & conversation_words)
            
            # Consider useful if has indicators and some relevance
            if usefulness_score > 0 and common_words >= 1:
                useful_count += 1
            elif common_words >= 3:  # Or if highly relevant
                useful_count += 1
        
        return useful_count / len(questions)

class EvaluationPipeline:
    """Main evaluation pipeline with ground truth annotation support"""
    
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
        df.to_csv("lexicon_ground_truth_evaluation.csv", index=False)
        print(f"\n💾 Ground Truth Evaluation saved to: lexicon_ground_truth_evaluation.csv")
        print(f"   📊 Rows: {len(df)} | Columns: {len(df.columns)}")
        print(f"   🎯 Target User Focus + Ground Truth Annotations")
        print(f"   🔥 Context Aggregation + Information Gap Detection Evaluation")

def main():
    """Main function for ground truth based evaluation"""
    
    # Dataset path - relative to current working directory
    DATASET_PATH = "./dataset"
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please ensure you're running this script from the correct directory.")
        return
    
    pipeline = EvaluationPipeline(DATASET_PATH)
    results = pipeline.run_complete_evaluation()
    
    print(f"\n🎉 Ground Truth Based evaluation complete!")
    print(f"✅ Successfully processed {len(results)} valid conversations")
    print(f"🎯 Target User Focus: Each conversation evaluated from single user perspective")
    print(f"🎲 Ground Truth: Reference resolution evaluated against manual annotations")
    print(f"🔥 Context Aggregation: Temporal/Spatial/Object reference disambiguation")
    print(f"❓ Information Gaps: Question usefulness and relevance evaluation")

if __name__ == "__main__":
    main()