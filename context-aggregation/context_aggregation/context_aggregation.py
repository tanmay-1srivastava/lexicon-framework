"""
Modular Context Aggregation Framework
Reads transcript files and uses real LLM APIs for processing
"""

import json
import os
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import AzureOpenAI
from pathlib import Path
from secret_keys import Open_ai_key

@dataclass
class SpeechData:
    transcript: str
    timestamp: str
    speaker: str
    confidence: float = 0.0

@dataclass
class ContextMetadata:
    location: str
    calendar: Dict
    participants: List[str]
    device_state: str
    relationships: Dict = None

@dataclass
class ContextOutput:
    resolved_content: Dict
    information_gaps: List[str]
    context_summary: Dict

class LLMClient:
    """Azure OpenAI client interface"""
    
    def __init__(self):
        # Azure OpenAI configuration
        self.endpoint = os.getenv("ENDPOINT_URL", "https://initial-resources.cognitiveservices.azure.com/")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
        self.subscription_key = Open_ai_key
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2025-01-01-preview",
        )
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text response from Azure OpenAI"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps with context aggregation and information processing."
                        }
                    ]
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            completion = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Azure OpenAI API Error: {e}")
            return ""
    
    def generate_json(self, prompt: str, max_tokens: int = 300) -> Dict:
        """Generate structured JSON response from Azure OpenAI"""
        json_prompt = prompt + "\n\nReturn your response as valid JSON only."
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text", 
                            "text": "You are an AI assistant that returns structured JSON responses for data processing."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json_prompt
                        }
                    ]
                }
            ]
            
            completion = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            result = completion.choices[0].message.content.strip()
            
            # Extract JSON from response if it contains other text
            if '{' in result and '}' in result:
                start = result.find('{')
                end = result.rfind('}') + 1
                result = result[start:end]
            
            return json.loads(result)
        
        except (json.JSONDecodeError, Exception) as e:
            print(f"JSON parsing error: {e}")
            return {}

class TranscriptLoader:
    """Handles loading and parsing transcript files"""
    
    @staticmethod
    def load_transcript_file(file_path: str) -> str:
        """Load transcript from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading transcript: {e}")
            return ""
    
    @staticmethod
    def parse_transcript(transcript: str) -> SpeechData:
        """Parse raw transcript into structured data"""
        # Remove silence markers and clean
        clean_transcript = transcript.replace("[silence]", " ").strip()
        clean_transcript = " ".join(clean_transcript.split())
        
        return SpeechData(
            transcript=clean_transcript,
            timestamp=datetime.datetime.now().isoformat(),
            speaker="User",
            confidence=1.0
        )

class ContextLoader:
    """Handles loading context metadata"""
    
    @staticmethod
    def load_context_file(file_path: str) -> ContextMetadata:
        """Load context metadata from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return ContextMetadata(
                location=data.get('location', 'Unknown'),
                calendar=data.get('calendar', {}),
                participants=data.get('participants', []),
                device_state=data.get('device_state', 'normal'),
                relationships=data.get('relationships', {})
            )
        except Exception as e:
            print(f"Error loading context: {e}")
            return ContextMetadata(
                location="Unknown",
                calendar={},
                participants=[],
                device_state="normal"
            )

class ContextAggregator:
    """Main Context Aggregation processor"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def process_transcript_file(self, transcript_path: str, context_path: str) -> ContextOutput:
        """Process transcript and context files"""
        
        # Load data
        transcript_raw = TranscriptLoader.load_transcript_file(transcript_path)
        speech_data = TranscriptLoader.parse_transcript(transcript_raw)
        context_metadata = ContextLoader.load_context_file(context_path)
        
        return self.process(speech_data, context_metadata)
    
    def process(self, speech_data: SpeechData, context_metadata: ContextMetadata) -> ContextOutput:
        """Main processing pipeline"""
        
        print("Processing Context Aggregation...")
        
        # Step 1: Reference Resolution
        resolved_content = self._resolve_references(speech_data, context_metadata)
        print("✓ References resolved")
        
        # Step 2: Context Enhancement  
        enhanced_content = self._enhance_context(resolved_content, context_metadata)
        print("✓ Context enhanced")
        
        # Step 3: Gap Detection
        information_gaps = self._find_gaps(enhanced_content)
        print("✓ Information gaps identified")
        
        # Step 4: Context Summary
        context_summary = self._create_summary(enhanced_content, context_metadata)
        print("✓ Context summary created")

        # print actual _resolve_references, _enhance_context, _find_gaps, _create_summary results for debugging
        
        return ContextOutput(
            resolved_content=enhanced_content,
            information_gaps=information_gaps,
            context_summary=context_summary
        )
    
    def _resolve_references(self, speech_data: SpeechData, context: ContextMetadata) -> Dict:
        """Resolve vague references using LLM"""
        
        prompt = f"""
        Original conversation: "{speech_data.transcript}"
        
        Context information:
        - Current location: {context.location}
        - Participants: {context.participants}
        - Current meeting/topic: {context.calendar.get('current_meeting', 'Unknown')}
        - Time context: {speech_data.timestamp}
        
        Task: Replace vague references with specific details:
        - Replace "this" with the specific project/topic
        - Replace "here" with the actual location  
        - Replace "tomorrow/today" with actual dates
        - Replace "we/us" with specific people
        
        Return only the conversation with references resolved.
        """
        
        resolved_speech = self.llm.generate(prompt)
        
        return {
            "original_speech": speech_data.transcript,
            "resolved_speech": resolved_speech,
            "timestamp": speech_data.timestamp,
            "location": context.location,
            "participants": context.participants,
            "meeting_context": context.calendar.get('current_meeting')
        }
    
    def _enhance_context(self, resolved_content: Dict, context: ContextMetadata) -> Dict:
        """Extract structured information using LLM"""
        
        prompt = f"""
        Conversation: "{resolved_content['resolved_speech']}"
        Location: {resolved_content['location']}
        Participants: {resolved_content['participants']}
        
        Extract the following information:
        {{
            "action_items": ["list of action items mentioned"],
            "topics": ["main topics discussed"], 
            "urgency_level": "low/medium/high",
            "meeting_type": "type of interaction",
            "main_purpose": "primary goal of conversation"
        }}
        """
        
        enhancement = self.llm.generate_json(prompt)
        
        # Merge with resolved content
        return {**resolved_content, **enhancement}
    
    def _find_gaps(self, enhanced_content: Dict) -> List[str]:
        """Identify information gaps using LLM"""
        
        prompt = f"""
        Conversation: "{enhanced_content.get('resolved_speech')}"
        Action items: {enhanced_content.get('action_items', [])}
        Participants: {enhanced_content.get('participants', [])}
        Purpose: {enhanced_content.get('main_purpose')}
        
        What specific information is missing to complete these action items?
        What questions remain unanswered?
        What details would help accomplish the stated goals?
        
        Return a JSON list of specific information gaps:
        ["gap1", "gap2", "gap3"]
        """
        
        try:
            response = self.llm.generate(prompt)
            # Extract list from response
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                gaps_json = response[start:end]
                return json.loads(gaps_json)
            else:
                # Fallback: split by lines/commas
                return [gap.strip() for gap in response.split('\n') if gap.strip()]
        except:
            return ["Unable to identify gaps"]
    
    def _create_summary(self, enhanced_content: Dict, context: ContextMetadata) -> Dict:
        """Create structured context summary"""
        
        return {
            "conversation_type": enhanced_content.get('meeting_type', 'discussion'),
            "urgency_level": enhanced_content.get('urgency_level', 'medium'),
            "location_context": enhanced_content['location'], 
            "temporal_context": enhanced_content['timestamp'],
            "participant_count": len(enhanced_content.get('participants', [])),
            "requires_collaboration": bool(enhanced_content.get('action_items')),
            "main_topics": enhanced_content.get('topics', []),
            "completeness_score": self._calculate_completeness(enhanced_content)
        }
    
    def _calculate_completeness(self, content: Dict) -> float:
        """Calculate how complete the context information is"""
        required_fields = ['resolved_speech', 'participants', 'action_items', 'main_purpose']
        completed = sum(1 for field in required_fields if content.get(field))
        return completed / len(required_fields)

def create_sample_files():
    """Create sample transcript and context files for testing"""
    
    # Sample transcript
    transcript_content = """
    Hi Sarah [silence] good thanks [silence] can we meet to discuss this tomorrow? [silence] 
    yes that would be great [silence] I think we need to finalize the design before the deadline [silence]
    absolutely, should we invite Mike too? [silence] good idea, I'll send him the meeting details
    """
    
    with open('sample_transcript.txt', 'w') as f:
        f.write(transcript_content)
    
    # Sample context
    context_content = {
        "location": "Conference Room B",
        "calendar": {
            "current_meeting": "Website Redesign Planning",
            "next_availability": "2024-11-11T14:00:00",
            "deadline": "2024-11-15T17:00:00"
        },
        "participants": ["User", "Sarah"],
        "device_state": "work_mode",
        "relationships": {
            "Sarah": "colleague",
            "Mike": "manager"
        }
    }
    
    with open('sample_context.json', 'w') as f:
        json.dump(context_content, f, indent=2)
    
    print("Sample files created: sample_transcript.txt, sample_context.json")

def main():
    """Main execution function"""
    
    # Check for Azure OpenAI API key
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    if not api_key or api_key == "REPLACE_WITH_YOUR_KEY_VALUE_HERE":
        print("Error: Please set AZURE_OPENAI_API_KEY environment variable")
        print("Also ensure ENDPOINT_URL and DEPLOYMENT_NAME are set if different from defaults")
        return
    
    # Create sample files if they don't exist
    if not os.path.exists('sample_transcript.txt'):
        create_sample_files()
    
    # Initialize components - no parameters needed now
    llm_client = LLMClient()
    aggregator = ContextAggregator(llm_client)
    
    # Process files
    try:
        result = aggregator.process_transcript_file('sample_transcript.txt', 'sample_context.json', degbug=True)
        
        print("\n" + "="*60)
        print("CONTEXT AGGREGATION RESULTS")
        print("="*60)
        
        print(f"\nORIGINAL: {result.resolved_content['original_speech']}")
        print(f"RESOLVED: {result.resolved_content['resolved_speech']}")
        
        print(f"\nACTION ITEMS: {result.resolved_content.get('action_items', [])}")
        print(f"TOPICS: {result.resolved_content.get('topics', [])}")
        print(f"URGENCY: {result.resolved_content.get('urgency_level', 'unknown')}")
        
        print(f"\nINFORMATION GAPS:")
        for gap in result.information_gaps:
            print(f"  - {gap}")
        
        print(f"\nCONTEXT SUMMARY:")
        for key, value in result.context_summary.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()