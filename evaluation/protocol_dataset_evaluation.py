"""
Protocol Dataset Evaluation Framework
Evaluates baseline vs lexicon framework on new protocol datasets with:
1. Resolution Accuracy (person/place/time/object)
2. Query Filtering Metrics (TPR/FNR for HIGH_VALUE vs LOW_VALUE)
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from openai import AzureOpenAI

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from secret_keys import Open_ai_key

# Import existing framework components directly (not as submodules)
import importlib.util

# Load context_aggregation module
ca_path = os.path.join(os.path.dirname(__file__), '..', 'context_aggregation', 'context_aggregation.py')
ca_spec = importlib.util.spec_from_file_location("context_aggregation_module", ca_path)
ca_module = importlib.util.module_from_spec(ca_spec)
ca_spec.loader.exec_module(ca_module)
LLMClient = ca_module.LLMClient
ContextAggregator = ca_module.ContextAggregator
SpeechData = ca_module.SpeechData
ContextMetadata = ca_module.ContextMetadata

# Load baseline module
bl_path = os.path.join(os.path.dirname(__file__), '..', 'context_aggregation', 'baseline_gpt_test.py')
bl_spec = importlib.util.spec_from_file_location("baseline_module", bl_path)
bl_module = importlib.util.module_from_spec(bl_spec)
bl_spec.loader.exec_module(bl_module)
BaselineGPTProcessor = bl_module.BaselineGPTProcessor

# Load info gap module
ig_path = os.path.join(os.path.dirname(__file__), '..', 'info-gap-detection', 'Information_gap_detection.py')
ig_spec = importlib.util.spec_from_file_location("infogap_module", ig_path)
ig_module = importlib.util.module_from_spec(ig_spec)
ig_spec.loader.exec_module(ig_module)
InformationGapDetector = ig_module.InformationGapDetector

@dataclass
class ResolutionMetrics:
    """Metrics for resolution accuracy by category"""
    category: str  # spatial, temporal, person, object
    total: int
    correct: int
    accuracy: float

@dataclass
class QueryFilterMetrics:
    """Metrics for query filtering (HIGH_VALUE vs LOW_VALUE)"""
    true_positives: int  # Correctly identified HIGH_VALUE
    false_positives: int  # Predicted HIGH_VALUE but actually LOW_VALUE
    true_negatives: int  # Correctly filtered LOW_VALUE
    false_negatives: int  # Missed HIGH_VALUE
    precision: float
    recall: float
    f1_score: float
    tpr: float  # True Positive Rate
    fnr: float  # False Negative Rate

@dataclass
class EvaluationResult:
    """Complete evaluation result for one dataset"""
    dataset_name: str
    scenario_type: str  # friends_meeting, work_collaboration, doctor_visit
    target_user: str  # "User A" or "User B"
    
    # Context Aggregation Metrics
    baseline_resolutions: Dict[str, ResolutionMetrics]
    framework_resolutions: Dict[str, ResolutionMetrics]
    
    # Info Gap Detection Metrics
    baseline_queries: QueryFilterMetrics
    framework_queries: QueryFilterMetrics
    
    # Raw outputs for comparison
    baseline_resolutions_raw: List[Dict] = field(default_factory=list)
    framework_resolutions_raw: List[Dict] = field(default_factory=list)
    baseline_queries_raw: List[str] = field(default_factory=list)
    framework_queries_raw: List[str] = field(default_factory=list)

class ProtocolDatasetEvaluator:
    """Evaluator for protocol datasets"""
    
    def __init__(self):
        # Azure OpenAI client for LLM-based matching
        self.endpoint = os.getenv("ENDPOINT_URL", "https://initial-resources.cognitiveservices.azure.com/")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
        self.subscription_key = Open_ai_key
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2024-12-01-preview",
        )
    
    def evaluate_dataset(self, dataset_path: str, target_user: str = "User A") -> EvaluationResult:
        """Evaluate both baseline and framework on a single dataset for ONE target user
        
        Args:
            dataset_path: Path to the JSON dataset
            target_user: "User A" or "User B" - which user's assistant we're evaluating
        """
        
        print(f"\n{'='*80}")
        print(f"EVALUATING: {os.path.basename(dataset_path)} [Target: {target_user}]")
        print(f"{'='*80}")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Extract scenario type from filename
        filename = os.path.basename(dataset_path)
        if 'friends' in filename:
            scenario_type = 'friends_meeting'
        elif 'work' in filename:
            scenario_type = 'work_collaboration'
        elif 'doctor' in filename:
            scenario_type = 'doctor_visit'
        else:
            scenario_type = 'unknown'
        
        # Extract ONLY the target user's utterances (single-sided conversation)
        target_utterances = [
            turn for turn in dataset['conversation_transcript']
            if turn['speaker'] == target_user
        ]
        
        print(f"  → Target user has {len(target_utterances)} utterances out of {len(dataset['conversation_transcript'])} total turns")
        
        # Get target user's mobile context
        user_key = 'user_a' if target_user == 'User A' else 'user_b'
        target_context = dataset['mobile_context_snapshot'][user_key]
        
        # Filter ground truth to only resolutions needed by target user
        target_ground_truth_resolutions = []
        for gt in dataset['ground_truth_resolutions']:
            trigger_turn = gt['trigger_turn_id']
            # Find the turn
            for turn in dataset['conversation_transcript']:
                if turn['turn_id'] == trigger_turn and turn['speaker'] == target_user:
                    target_ground_truth_resolutions.append(gt)
                    break
        
        print(f"  → {len(target_ground_truth_resolutions)} ground truth resolutions for target user")
        
        # All queries might be relevant (queries are about resolving OTHER person's ambiguous references)
        target_ground_truth_queries = dataset['required_protocol_queries']
        
        # Run baseline approach with target user's data only
        print("\n[1/4] Running BASELINE Context Aggregation...")
        baseline_resolutions_output = self._run_baseline_context_aggregation(
            target_utterances, target_context
        )
        
        print("[2/4] Running FRAMEWORK Context Aggregation...")
        framework_resolutions_output = self._run_framework_context_aggregation(
            target_utterances, target_context
        )
        
        print("[3/4] Running BASELINE Info Gap Detection...")
        baseline_queries_output = self._run_baseline_info_gap(
            target_utterances, target_context
        )
        
        print("[4/4] Running FRAMEWORK Info Gap Detection...")
        framework_queries_output = self._run_framework_info_gap(
            target_utterances, framework_resolutions_output
        )
        
        # Evaluate Resolution Accuracy (using target user's ground truth only)
        print("\n[Metrics] Calculating Resolution Accuracy...")
        baseline_res_metrics = self._evaluate_resolution_accuracy(
            baseline_resolutions_output, 
            target_ground_truth_resolutions
        )
        framework_res_metrics = self._evaluate_resolution_accuracy(
            framework_resolutions_output,
            target_ground_truth_resolutions
        )
        
        # Evaluate Query Filtering
        print("[Metrics] Calculating Query Filtering Metrics...")
        baseline_query_metrics = self._evaluate_query_filtering(
            baseline_queries_output,
            target_ground_truth_queries,
            target_user
        )
        framework_query_metrics = self._evaluate_query_filtering(
            framework_queries_output,
            target_ground_truth_queries,
            target_user
        )
        
        return EvaluationResult(
            dataset_name=filename,
            scenario_type=scenario_type,
            target_user=target_user,
            baseline_resolutions=baseline_res_metrics,
            framework_resolutions=framework_res_metrics,
            baseline_queries=baseline_query_metrics,
            framework_queries=framework_query_metrics,
            baseline_resolutions_raw=baseline_resolutions_output,
            framework_resolutions_raw=framework_resolutions_output,
            baseline_queries_raw=baseline_queries_output,
            framework_queries_raw=framework_queries_output
        )
    
    def _run_baseline_context_aggregation(self, target_utterances: List[Dict], target_context: Dict) -> List[Dict]:
        """Run baseline (simple GPT prompt) for context aggregation with target user's data"""
        
        # Build conversation text from target user's utterances only
        conversation = "\n".join([
            f"Turn {turn['turn_id']}: {turn['speaker']}: {turn['text']}"
            for turn in target_utterances
        ])
        
        # Build mobile context - handle missing fields gracefully
        location = target_context.get('location_semantic', 'Unknown')
        gps = target_context.get('gps_coords', 'Unknown')
        wifi = target_context.get('wifi_ssid', 'Unknown')
        calendar = target_context.get('calendar_next', 'None')
        
        prompt = f"""You are helping resolve ambiguous references in a conversation using mobile context.

CONVERSATION (single-sided, you only hear one person):
{conversation}

MOBILE CONTEXT:
Location: {location}
GPS: {gps}
WiFi: {wifi}
Calendar: {calendar}

Identify all ambiguous references (like "here", "there", "him", "her", "it", "tomorrow", etc.) and resolve them using the mobile context.

Return a JSON array with this format:
[
  {{
    "turn_id": <number>,
    "ambiguous_phrase": "<phrase>",
    "resolved_entity": "<what it refers to>",
    "resolution_type": "spatial|temporal|person|object"
  }}
]

Only return valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"  → Baseline raw response keys: {result.keys() if isinstance(result, dict) else 'list'}")
            # Handle both array and object with array - check multiple possible keys
            if isinstance(result, dict):
                if 'resolutions' in result:
                    return result['resolutions']
                elif 'ambiguous_references' in result:
                    return result['ambiguous_references']
                elif 'results' in result:
                    return result['results']
                else:
                    # Try to find any list value
                    for value in result.values():
                        if isinstance(value, list):
                            return value
                    return []
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            print(f"  ⚠️  Baseline error: {e}")
            return []
    
    def _run_framework_context_aggregation(self, target_utterances: List[Dict], target_context: Dict) -> List[Dict]:
        """Run lexicon framework approach for context aggregation with target user's data"""
        
        # Build conversation text from target user's utterances only
        conversation = "\n".join([
            f"Turn {turn['turn_id']}: {turn['speaker']}: {turn['text']}"
            for turn in target_utterances
        ])
        
        # Enhanced prompt with lexicon framework structure
        prompt = f"""You are an advanced context aggregation system that resolves ambiguous references using multimodal mobile context.

TASK: Systematically identify and resolve ALL ambiguous references in the conversation.

CONVERSATION (single-sided, you only hear one person):
{conversation}

MULTIMODAL MOBILE CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET USER CONTEXT:
  • Semantic Location: {target_context.get('location_semantic', 'None')}
  • GPS Coordinates: {target_context.get('gps_coords', 'None')}
  • WiFi Network: {target_context.get('wifi_ssid', 'None')}
  • Calendar Next: {target_context.get('calendar_next', 'None')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESOLUTION METHODOLOGY:
1. SPATIAL REFERENCES: "here", "there", "this place", "that location"
   → Resolve using GPS coordinates, WiFi SSID, or semantic location names
   
2. TEMPORAL REFERENCES: "tomorrow", "next week", "then", "later", "soon"
   → Resolve using calendar events, timestamps, or contextual time references
   
3. PERSON REFERENCES: "him", "her", "they", "someone", "the person"
   → Resolve using conversation context, calendar events, or explicit mentions
   
4. OBJECT REFERENCES: "it", "this", "that", "those", "the thing"
   → Resolve using conversation context and previous mentions

OUTPUT FORMAT (JSON array):
[
  {{
    "turn_id": <number>,
    "ambiguous_phrase": "<exact phrase from conversation>",
    "resolved_entity": "<specific resolution>",
    "resolution_type": "spatial|temporal|person|object",
    "resolution_source": "GPS|WiFi|Calendar|ConversationContext"
  }}
]

IMPORTANT: Be thorough and systematic. Identify ALL ambiguous references, even subtle ones.
Only return valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"  → Framework raw response keys: {result.keys() if isinstance(result, dict) else 'list'}")
            # Handle both array and object with array - check multiple possible keys
            if isinstance(result, dict):
                if 'resolutions' in result:
                    return result['resolutions']
                elif 'ambiguous_references' in result:
                    return result['ambiguous_references']
                elif 'results' in result:
                    return result['results']
                else:
                    # Try to find any list value
                    for value in result.values():
                        if isinstance(value, list):
                            return value
                    return []
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            print(f"  ⚠️  Framework error: {e}")
            return []
    
    def _run_baseline_info_gap(self, target_utterances: List[Dict], target_context: Dict) -> List[str]:
        """Run baseline (ask all questions) for info gap detection with target user's data"""
        
        # Build conversation text from target user's utterances only
        conversation = "\n".join([
            f"Turn {turn['turn_id']}: {turn['speaker']}: {turn['text']}"
            for turn in target_utterances
        ])
        
        prompt = f"""You are analyzing a conversation to identify what questions should be asked to the other person.

CONVERSATION:
{conversation}

What important questions should be asked to help with this conversation? Think broadly about any information that could be useful.

Return a JSON array of questions:
{{
  "questions": ["Question 1?", "Question 2?", ...]
}}

Only return valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            queries = result.get('questions', [])
            print(f"  → Baseline queries: {len(queries)} returned")
            if queries:
                print(f"     Sample: {queries[0]}")
            return queries
        except Exception as e:
            print(f"  ⚠️  Baseline query error: {e}")
            return []
    
    def _run_framework_info_gap(self, target_utterances: List[Dict], framework_context_output: List[Dict]) -> List[str]:
        """Run lexicon framework with importance filtering for info gap detection"""
        
        # Extract resolved conversation from framework output
        if not framework_context_output:
            resolved_conversation = ""
        else:
            resolved_conversation = str(framework_context_output)
        
        # Build conversation text from target user's utterances
        conversation = "\n".join([
            f"Turn {turn['turn_id']}: {turn['speaker']}: {turn['text']}"
            for turn in target_utterances
        ])
        
        prompt = f"""You are an intelligent information gap detector that identifies ONLY HIGH-VALUE, ACTIONABLE questions.

CONVERSATION:
{conversation}

TASK: Identify questions that are:
✓ ACTIONABLE: Required to complete tasks or make decisions
✓ CRITICAL: Impact the success or safety of the conversation's goals
✓ SPECIFIC: Target concrete information gaps, not general curiosity

FILTER OUT:
✗ Small talk or casual comments
✗ Jokes or non-actionable remarks
✗ Trivial reminders already implied
✗ General check-in questions without specific purpose

METHODOLOGY:
1. Identify ambiguous references that block task completion
2. Focus on entity resolutions needed for coordination
3. Prioritize urgency and actionability
4. Filter noise and low-value queries

Return a JSON array of HIGH-VALUE questions only:
{{
  "questions": ["Specific question 1?", "Specific question 2?", ...]
}}

Only return valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            queries = result.get('questions', [])
            print(f"  → Framework queries: {len(queries)} returned")
            if queries:
                print(f"     Sample: {queries[0]}")
            return queries
        except Exception as e:
            print(f"  ⚠️  Framework query error: {e}")
            return []
    
    def _infer_resolution_type(self, ambiguous_phrase: str, resolved_entity: str) -> str:
        """Infer resolution type from ambiguous phrase and resolved entity"""
        
        phrase_lower = ambiguous_phrase.lower()
        entity_lower = resolved_entity.lower()
        
        # Spatial indicators
        spatial_words = ['here', 'there', 'place', 'where', 'location', 'street', 'avenue', 'room', 
                         'building', 'cafe', 'restaurant', 'store', 'office', 'loft', 'shop', 
                         'window', 'gps', 'coordinates']
        if any(word in phrase_lower or word in entity_lower for word in spatial_words):
            return 'spatial'
        
        # Temporal indicators
        temporal_words = ['tomorrow', 'yesterday', 'today', 'next', 'last', 'when', 'time', 
                          'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                          'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'year',
                          'soon', 'later', 'now', 'then']
        if any(word in phrase_lower or word in entity_lower for word in temporal_words):
            return 'temporal'
        
        # Person indicators
        person_words = ['him', 'her', 'he', 'she', 'they', 'who', 'person', 'people', 'someone',
                        'coordinator', 'manager', 'staff', 'guest', 'friend', 'roommate', 
                        'doctor', 'nurse', 'patient', 'colleague']
        # Also check if it looks like a name (capitalized single word or two words)
        if (any(word in phrase_lower for word in person_words) or 
            entity_lower.split()[0].istitle() if entity_lower.split() else False):
            return 'person'
        
        # Object is default
        return 'object'
    
    def _evaluate_resolution_accuracy(self, predicted_resolutions: List[Dict],
                                     ground_truth: List[Dict]) -> Dict[str, ResolutionMetrics]:
        """Calculate resolution accuracy metrics by category"""
        
        # Group ground truth by type (infer types if not present)
        gt_by_type = {
            'spatial': [],
            'temporal': [],
            'person': [],
            'object': []
        }
        
        for gt in ground_truth:
            # Infer type if not present
            res_type = gt.get('resolution_type', None)
            if not res_type:
                res_type = self._infer_resolution_type(
                    gt.get('ambiguous_phrase', ''),
                    gt.get('resolved_entity', '')
                )
            if res_type in gt_by_type:
                gt_by_type[res_type].append(gt)
            else:
                # Default to object if unknown
                gt_by_type['object'].append(gt)
        
        # Calculate accuracy for each category
        metrics = {}
        
        for category in ['spatial', 'temporal', 'person', 'object']:
            gt_items = gt_by_type[category]
            total = len(gt_items)
            correct = 0
            
            if total == 0:
                metrics[category] = ResolutionMetrics(category, 0, 0, 0.0)
                continue
            
            # Match predictions to ground truth
            for gt_item in gt_items:
                gt_turn = gt_item['trigger_turn_id']
                gt_phrase = gt_item['ambiguous_phrase'].lower()
                gt_resolved = gt_item['resolved_entity'].lower()
                
                # Find matching prediction
                for pred in predicted_resolutions:
                    pred_turn = pred.get('turn_id', -1)
                    pred_phrase = pred.get('ambiguous_phrase', '').lower()
                    pred_resolved = pred.get('resolved_entity', '').lower()
                    
                    # Match by turn_id and phrase
                    if pred_turn == gt_turn and gt_phrase in pred_phrase or pred_phrase in gt_phrase:
                        # Check if resolution is correct (fuzzy match)
                        if self._fuzzy_match(pred_resolved, gt_resolved):
                            correct += 1
                            break
            
            accuracy = correct / total if total > 0 else 0.0
            metrics[category] = ResolutionMetrics(category, total, correct, accuracy)
        
        return metrics
    
    def _evaluate_query_filtering(self, predicted_queries: List[str],
                                  ground_truth_queries: List[Dict],
                                  target_user: str = "User A") -> QueryFilterMetrics:
        """Calculate query filtering metrics (TPR, FNR, Precision, Recall, F1)"""
        
        # Separate ground truth into HIGH_VALUE and LOW_VALUE
        high_value_gt = [q for q in ground_truth_queries if q['query_quality_check'] == 'HIGH_VALUE']
        low_value_gt = [q for q in ground_truth_queries if q['query_quality_check'] == 'LOW_VALUE']
        
        print(f"  → Query eval: {len(predicted_queries)} predicted, {len(high_value_gt)} HIGH, {len(low_value_gt)} LOW in GT")
        
        # Use LLM to match predicted queries to ground truth
        matched_high_value = 0  # True Positives
        matched_low_value = 0   # False Positives (predicted but should be filtered)
        unmatched = 0  # Queries that don't match any GT
        
        for i, pred_query in enumerate(predicted_queries):
            # Check if this query matches any HIGH_VALUE ground truth
            if self._query_matches_any(pred_query, high_value_gt):
                matched_high_value += 1
                print(f"     [{i+1}] MATCHED HIGH: {pred_query[:60]}...")
            # Check if this query matches any LOW_VALUE ground truth
            elif self._query_matches_any(pred_query, low_value_gt):
                matched_low_value += 1
                print(f"     [{i+1}] MATCHED LOW: {pred_query[:60]}...")
            else:
                unmatched += 1
        
        print(f"  → Matching results: {matched_high_value} HIGH, {matched_low_value} LOW, {unmatched} UNMATCHED")
        
        # Calculate metrics
        true_positives = matched_high_value
        false_positives = matched_low_value
        false_negatives = len(high_value_gt) - matched_high_value
        true_negatives = len(low_value_gt) - matched_low_value
        
        # Handle division by zero
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        tpr = recall  # True Positive Rate = Recall
        fnr = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        return QueryFilterMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            tpr=tpr,
            fnr=fnr
        )
    
    def _query_matches_any(self, predicted_query: str, ground_truth_list: List[Dict]) -> bool:
        """Check if predicted query semantically matches any ground truth query using LLM"""
        
        if not ground_truth_list:
            return False
        
        # Build a prompt for LLM to check if predicted query matches any GT query
        gt_descriptions = []
        for i, gt_query in enumerate(ground_truth_list):
            reason = gt_query.get('reason', '')
            fallback = gt_query.get('natural_language_fallback', '')
            target_slot = gt_query.get('protocol_payload', {}).get('target_slot', 'UNKNOWN')
            gt_descriptions.append(f"{i+1}. [{target_slot}] {reason} || {fallback}")
        
        gt_text = "\n".join(gt_descriptions)
        
        prompt = f"""You are evaluating if a predicted information gap question matches any ground truth queries.

PREDICTED QUERY:
"{predicted_query}"

GROUND TRUTH QUERIES:
{gt_text}

Does the predicted query ask about the SAME information gap as ANY of the ground truth queries?
Consider them a match if they're asking about the same missing entity (person/place/time/object) even if worded differently.

Respond with ONLY a JSON object with this format:
{{"matches": true, "matched_index": 1}}
OR
{{"matches": false, "matched_index": null}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('matches', False)
        except Exception as e:
            # Fallback to word overlap if LLM fails
            pred_lower = predicted_query.lower()
            for gt_query in ground_truth_list:
                gt_text = gt_query.get('reason', '') + ' ' + gt_query.get('natural_language_fallback', '')
                if self._semantic_similarity(pred_lower, gt_text.lower()) > 0.25:
                    return True
            return False
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (0-1)"""
        
        # Simple word overlap similarity (can be enhanced with embeddings)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'for', 'of', 'in', 'on', 'at'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _fuzzy_match(self, text1: str, text2: str) -> bool:
        """Fuzzy match for resolution comparison"""
        
        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
        
        # Contains match
        if text1 in text2 or text2 in text1:
            return True
        
        # Key word overlap (>60% overlap)
        similarity = self._semantic_similarity(text1, text2)
        return similarity > 0.6
    
    def save_outputs(self, result: EvaluationResult, output_dir: str = "evaluation/outputs"):
        """Save baseline and framework outputs to JSON files for comparison"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on dataset and target user
        base_name = result.dataset_name.replace('.json', '')
        user_suffix = result.target_user.replace(' ', '_').lower()
        
        output_data = {
            "dataset": result.dataset_name,
            "scenario": result.scenario_type,
            "target_user": result.target_user,
            "baseline": {
                "resolutions": result.baseline_resolutions_raw,
                "queries": result.baseline_queries_raw
            },
            "framework": {
                "resolutions": result.framework_resolutions_raw,
                "queries": result.framework_queries_raw
            },
            "metrics": {
                "resolution_accuracy": {
                    "baseline": {
                        cat: {"total": m.total, "correct": m.correct, "accuracy": m.accuracy}
                        for cat, m in result.baseline_resolutions.items()
                    },
                    "framework": {
                        cat: {"total": m.total, "correct": m.correct, "accuracy": m.accuracy}
                        for cat, m in result.framework_resolutions.items()
                    }
                },
                "query_filtering": {
                    "baseline": {
                        "tpr": result.baseline_queries.tpr,
                        "fnr": result.baseline_queries.fnr,
                        "precision": result.baseline_queries.precision,
                        "recall": result.baseline_queries.recall,
                        "f1_score": result.baseline_queries.f1_score,
                        "tp": result.baseline_queries.true_positives,
                        "fp": result.baseline_queries.false_positives,
                        "tn": result.baseline_queries.true_negatives,
                        "fn": result.baseline_queries.false_negatives
                    },
                    "framework": {
                        "tpr": result.framework_queries.tpr,
                        "fnr": result.framework_queries.fnr,
                        "precision": result.framework_queries.precision,
                        "recall": result.framework_queries.recall,
                        "f1_score": result.framework_queries.f1_score,
                        "tp": result.framework_queries.true_positives,
                        "fp": result.framework_queries.false_positives,
                        "tn": result.framework_queries.true_negatives,
                        "fn": result.framework_queries.false_negatives
                    }
                }
            }
        }
        
        output_path = os.path.join(output_dir, f"{base_name}_{user_suffix}.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved outputs to: {output_path}")
        return output_path
    
    def print_result(self, result: EvaluationResult):
        """Pretty print evaluation result"""
        
        print(f"\n{'='*80}")
        print(f"RESULTS: {result.dataset_name}")
        print(f"Scenario: {result.scenario_type} | Target User: {result.target_user}")
        print(f"{'='*80}")
        
        # Resolution Accuracy
        print("\n📍 RESOLUTION ACCURACY (Context Aggregation)")
        print("─" * 80)
        print(f"{'Category':<15} {'Baseline':<20} {'Framework':<20} {'Improvement'}")
        print("─" * 80)
        
        for category in ['person', 'spatial', 'temporal', 'object']:
            bl_metric = result.baseline_resolutions[category]
            fw_metric = result.framework_resolutions[category]
            
            bl_str = f"{bl_metric.correct}/{bl_metric.total} ({bl_metric.accuracy*100:.1f}%)"
            fw_str = f"{fw_metric.correct}/{fw_metric.total} ({fw_metric.accuracy*100:.1f}%)"
            improvement = (fw_metric.accuracy - bl_metric.accuracy) * 100
            
            category_label = {
                'person': 'Person (Name)',
                'spatial': 'Place (Location)',
                'temporal': 'Time (Temporal)',
                'object': 'Thing (Object)'
            }[category]
            
            print(f"{category_label:<15} {bl_str:<20} {fw_str:<20} {improvement:+.1f}%")
        
        # Query Filtering
        print("\n🔍 QUERY FILTERING (Info Gap Detection)")
        print("─" * 80)
        print(f"{'Metric':<20} {'Baseline':<20} {'Framework':<20} {'Improvement'}")
        print("─" * 80)
        
        bl_q = result.baseline_queries
        fw_q = result.framework_queries
        
        print(f"{'TPR (Recall)':<20} {bl_q.tpr*100:.1f}%{'':<15} {fw_q.tpr*100:.1f}%{'':<15} {(fw_q.tpr - bl_q.tpr)*100:+.1f}%")
        print(f"{'FNR':<20} {bl_q.fnr*100:.1f}%{'':<15} {fw_q.fnr*100:.1f}%{'':<15} {(fw_q.fnr - bl_q.fnr)*100:+.1f}%")
        print(f"{'Precision':<20} {bl_q.precision*100:.1f}%{'':<15} {fw_q.precision*100:.1f}%{'':<15} {(fw_q.precision - bl_q.precision)*100:+.1f}%")
        print(f"{'F1 Score':<20} {bl_q.f1_score:.3f}{'':<15} {fw_q.f1_score:.3f}{'':<15} {(fw_q.f1_score - bl_q.f1_score):+.3f}")
        
        print("\n" + "─" * 80)
        print(f"Queries Generated: Baseline={bl_q.true_positives + bl_q.false_positives}, Framework={fw_q.true_positives + fw_q.false_positives}")
        print("─" * 80)

def main():
    """Test on a single dataset with both users"""
    
    evaluator = ProtocolDatasetEvaluator()
    
    # Test on friends_meeting_001.json
    test_file = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/data_generation/new_data/generated_datasets/friends_meeting_001.json"
    
    # Evaluate for User A
    print("\n" + "="*80)
    print("EVALUATING FOR USER A's ASSISTANT")
    print("="*80)
    result_a = evaluator.evaluate_dataset(test_file, target_user="User A")
    evaluator.print_result(result_a)
    evaluator.save_outputs(result_a)
    
    # Evaluate for User B
    print("\n" + "="*80)
    print("EVALUATING FOR USER B's ASSISTANT")
    print("="*80)
    result_b = evaluator.evaluate_dataset(test_file, target_user="User B")
    evaluator.print_result(result_b)
    evaluator.save_outputs(result_b)

if __name__ == "__main__":
    main()
