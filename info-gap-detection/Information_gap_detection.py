"""
Information Gap Detection Framework
Takes Context Aggregation output and identifies important questions for A2A communication
Uses real Azure OpenAI calls with simulated realistic data
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import AzureOpenAI
from secret_keys import Open_ai_key

@dataclass
class ImportantQuestion:
    question: str
    importance_score: float
    question_type: str
    topic: str
    reasoning: str

class InformationGapDetector:
    """Detects important information gaps from Context Aggregation output"""
    
    def __init__(self):
        # Azure OpenAI configuration (same as Context Aggregation)
        self.endpoint = os.getenv("ENDPOINT_URL", "https://initial-resources.cognitiveservices.azure.com/")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
        self.subscription_key = Open_ai_key
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2025-01-01-preview",
        )
    
    def detect_important_gaps(self, context_aggregation_output: Dict) -> List[ImportantQuestion]:
        """Main pipeline: Context Aggregation Output → Important Questions"""
        
        print("🔍 Detecting Important Information Gaps...")
        
        # Extract relevant data from Context Aggregation output
        resolved_content = context_aggregation_output['resolved_content']
        existing_gaps = context_aggregation_output['information_gaps']
        context_summary = context_aggregation_output['context_summary']
        
        # Step 1: Identify ALL potentially useful information
        all_potential_gaps = self._identify_all_gaps(resolved_content, existing_gaps, context_summary)
        
        # Step 2: Extract only IMPORTANT questions
        important_questions = self._extract_important_questions(all_potential_gaps, resolved_content)
        
        print(f"✓ Identified {len(important_questions)} important questions")
        return important_questions
    
    def _identify_all_gaps(self, resolved_content: Dict, existing_gaps: List[str], context_summary: Dict) -> List[Dict]:
        """Use LLM to identify ALL potentially useful information (comprehensive)"""
        
        prompt = f"""
        You are analyzing a conversation to identify ALL information that could potentially be useful from another person.
        
        CONVERSATION: "{resolved_content['resolved_speech']}"
        ACTION ITEMS: {resolved_content.get('action_items', [])}
        TOPICS: {resolved_content.get('topics', [])}
        CONVERSATION TYPE: {context_summary.get('conversation_type', 'unknown')}
        URGENCY: {context_summary.get('urgency_level', 'medium')}
        
        EXISTING GAPS ALREADY IDENTIFIED: {existing_gaps}
        
        Think broadly about what information from another person could be useful:
        1. TASK COMPLETION: Direct information needed to complete mentioned tasks
        2. CONTEXT ENHANCEMENT: Information that would improve understanding or decision-making  
        3. COORDINATION: Information needed for scheduling, planning, or collaboration
        4. BACKGROUND: Relevant experiences, opinions, or knowledge the other person might have
        5. FOLLOW-UP: Future-relevant information based on upcoming events or plans
        
        For each gap, provide:
        - What specific information is needed
        - Why it would be useful
        - How urgent/important it is
        - What category it falls under
        
        Return JSON format:
        [
            {{
                "information_needed": "specific information description",
                "reasoning": "why this would be useful",
                "importance": "high/medium/low", 
                "category": "task_completion/context_enhancement/coordination/background/follow_up",
                "urgency": "immediate/soon/eventual"
            }}
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an AI assistant that identifies comprehensive information gaps for agent-to-agent collaboration."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                max_tokens=2000,
                temperature=0.1,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            result = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if '[' in result and ']' in result:
                start = result.find('[')
                end = result.rfind(']') + 1
                gaps_json = result[start:end]
                return json.loads(gaps_json)
            else:
                print("Could not parse LLM response as JSON")
                print(f"Response content: {result}")
                return []
                
        except Exception as e:
            print(f"Error in gap identification: {e}")
            return []
    
    def _extract_important_questions(self, all_gaps: List[Dict], resolved_content: Dict) -> List[ImportantQuestion]:
        """Filter to only IMPORTANT questions worth asking another agent"""
        
        if not all_gaps:
            return []
        
        prompt = f"""
        You have identified potential information gaps. Now filter to only the IMPORTANT ones worth asking another AI agent.
        
        ORIGINAL CONVERSATION: "{resolved_content['resolved_speech']}"
        
        POTENTIAL GAPS: {json.dumps(all_gaps, indent=2)}
        
        FILTERING CRITERIA - Only include questions that are:
        1. DIRECTLY RELEVANT to the conversation or immediate tasks
        2. ACTIONABLE - the answer would actually help accomplish something
        3. APPROPRIATE - reasonable to ask in this context
        4. SPECIFIC - not too vague or general
        
        EXCLUDE questions that are:
        - Too personal or invasive
        - Not directly related to the conversation  
        - Too vague or broad
        - Nice-to-know but not important
        
        For each IMPORTANT gap, convert it into a specific question and explain why it's important.
        
        Return JSON format:
        [
            {{
                "question": "Specific question to ask the other agent",
                "importance_score": 0.8,
                "question_type": "task_completion/coordination/context_enhancement", 
                "topic": "main topic of question",
                "reasoning": "why this question is important enough to ask"
            }}
        ]
        
        Only include questions you think are truly important - quality over quantity.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system", 
                        "content": [{"type": "text", "text": "You are an AI assistant that filters to only important, actionable questions for agent-to-agent collaboration."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                max_tokens=1500,
                temperature=0.1,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            result = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if '[' in result and ']' in result:
                start = result.find('[')
                end = result.rfind(']') + 1
                questions_json = result[start:end]
                questions_data = json.loads(questions_json)
                
                # Convert to ImportantQuestion objects
                important_questions = []
                for q_data in questions_data:
                    question = ImportantQuestion(
                        question=q_data.get('question', ''),
                        importance_score=q_data.get('importance_score', 0.5),
                        question_type=q_data.get('question_type', 'unknown'),
                        topic=q_data.get('topic', 'general'),
                        reasoning=q_data.get('reasoning', '')
                    )
                    important_questions.append(question)
                
                return important_questions
            else:
                print("Could not parse questions response as JSON")
                return []
                
        except Exception as e:
            print(f"Error in important question extraction: {e}")
            return []

def create_sample_context_aggregation_outputs():
    """Generate realistic Context Aggregation outputs for testing"""
    
    samples = [
        {
            "name": "Team Project Coordination",
            "output": {
                "resolved_content": {
                    "original_speech": "Hey Sarah can we meet about this project deadline tomorrow?",
                    "resolved_speech": "Hey Sarah, can we meet about the Website Redesign Project deadline tomorrow November 15th?",
                    "timestamp": "2024-11-14T14:30:00",
                    "location": "Office Floor 3", 
                    "participants": ["User", "Sarah"],
                    "meeting_context": "Website Redesign Planning",
                    "action_items": ["Schedule meeting about project deadline", "Discuss timeline for Website Redesign Project"],
                    "topics": ["Website Redesign Project", "deadline discussion", "team meeting"],
                    "urgency_level": "high",
                    "meeting_type": "urgent coordination",
                    "main_purpose": "Coordinate urgent project deadline discussion"
                },
                "information_gaps": [
                    "Sarah's availability tomorrow",
                    "Current project status and blockers", 
                    "Specific meeting time preference"
                ],
                "context_summary": {
                    "conversation_type": "urgent coordination",
                    "urgency_level": "high",
                    "location_context": "Office Floor 3",
                    "participant_count": 2,
                    "requires_collaboration": True,
                    "main_topics": ["Website Redesign Project", "deadline", "meeting"],
                    "completeness_score": 0.75
                }
            }
        },
        
        {
            "name": "Conference Travel Planning", 
            "output": {
                "resolved_content": {
                    "original_speech": "I'm planning to go to that conference in San Francisco next month",
                    "resolved_speech": "I'm planning to go to the AI Research Conference in San Francisco next month in December 2024",
                    "timestamp": "2024-11-14T16:20:00",
                    "location": "Home Office",
                    "participants": ["User", "Colleague"],
                    "meeting_context": None,
                    "action_items": ["Plan conference attendance", "Arrange travel to San Francisco"],
                    "topics": ["AI Research Conference", "San Francisco travel", "conference planning"],
                    "urgency_level": "medium",
                    "meeting_type": "informal planning",
                    "main_purpose": "Discuss upcoming conference attendance"
                },
                "information_gaps": [
                    "Conference registration details",
                    "Travel arrangements and accommodation",
                    "Colleague's attendance plans"
                ],
                "context_summary": {
                    "conversation_type": "informal planning",
                    "urgency_level": "medium", 
                    "location_context": "Home Office",
                    "participant_count": 2,
                    "requires_collaboration": True,
                    "main_topics": ["conference", "travel", "San Francisco"],
                    "completeness_score": 0.60
                }
            }
        },
        
        {
            "name": "Casual Restaurant Discussion",
            "output": {
                "resolved_content": {
                    "original_speech": "Do you know any good Italian restaurants around here?",
                    "resolved_speech": "Do you know any good Italian restaurants around the Financial District area?",
                    "timestamp": "2024-11-14T12:15:00",
                    "location": "Financial District",
                    "participants": ["User", "Coworker"],
                    "meeting_context": "Lunch break",
                    "action_items": ["Find Italian restaurant recommendation"],
                    "topics": ["Italian restaurants", "local dining", "restaurant recommendations"], 
                    "urgency_level": "low",
                    "meeting_type": "casual conversation",
                    "main_purpose": "Get restaurant recommendation for immediate dining"
                },
                "information_gaps": [
                    "Specific Italian restaurant recommendations",
                    "Restaurant preferences (price range, atmosphere)",
                    "Availability for immediate dining"
                ],
                "context_summary": {
                    "conversation_type": "casual conversation",
                    "urgency_level": "low",
                    "location_context": "Financial District", 
                    "participant_count": 2,
                    "requires_collaboration": False,
                    "main_topics": ["restaurants", "recommendations"],
                    "completeness_score": 0.80
                }
            }
        }
    ]
    
    return samples

def main():
    """Test Information Gap Detection with realistic data"""
    
    # Initialize detector
    gap_detector = InformationGapDetector()
    
    # Get sample Context Aggregation outputs
    samples = create_sample_context_aggregation_outputs()
    
    print("=" * 70)
    print("INFORMATION GAP DETECTION TESTING")
    print("=" * 70)
    
    for sample in samples:
        print(f"\n🔄 Processing: {sample['name']}")
        print("-" * 50)
        
        # Run Information Gap Detection
        important_questions = gap_detector.detect_important_gaps(sample['output'])
        
        # Display results
        print(f"\n📝 ORIGINAL CONVERSATION:")
        print(f"   {sample['output']['resolved_content']['original_speech']}")
        
        print(f"\n🎯 IMPORTANT QUESTIONS TO ASK OTHER AGENT:")
        if important_questions:
            for i, question in enumerate(important_questions, 1):
                print(f"   {i}. {question.question}")
                print(f"      → Type: {question.question_type} | Score: {question.importance_score:.2f}")
                print(f"      → Reasoning: {question.reasoning}")
                print()
        else:
            print("   No important questions identified")
        
        print("=" * 70)

if __name__ == "__main__":
    main()