"""
Protocol Dataset Evaluation Framework (Complete Version)
Evaluates baseline vs lexicon framework on protocol datasets.
"""

import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from openai import AzureOpenAI
import importlib.util

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Securely load your key
try:
    from secret_keys import Open_ai_key
except ImportError:
    Open_ai_key = "YOUR_KEY_HERE"

# --- DYNAMIC MODULE LOADING (Retained from your original structure) ---
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ca_path = os.path.join(os.path.dirname(__file__), '..', 'context_aggregation', 'context_aggregation.py')
ig_path = os.path.join(os.path.dirname(__file__), '..', 'info-gap-detection', 'Information_gap_detection.py')

# Load the logic if needed (Evaluation methods use direct API calls for strict control)
ca_module = load_module_from_path("context_aggregation_module", ca_path)
ig_module = load_module_from_path("infogap_module", ig_path)

@dataclass
class ResolutionMetrics:
    category: str
    total: int
    correct: int
    accuracy: float

@dataclass
class QueryFilterMetrics:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    tpr: float
    fnr: float

@dataclass
class EvaluationResult:
    dataset_name: str
    scenario_type: str
    target_user: str
    baseline_resolutions: Dict[str, ResolutionMetrics]
    framework_resolutions: Dict[str, ResolutionMetrics]
    baseline_queries: QueryFilterMetrics
    framework_queries: QueryFilterMetrics
    baseline_resolutions_raw: List[Dict] = field(default_factory=list)
    framework_resolutions_raw: List[Dict] = field(default_factory=list)
    baseline_queries_raw: List[str] = field(default_factory=list)
    framework_queries_raw: List[str] = field(default_factory=list)

class ProtocolDatasetEvaluator:
    def __init__(self):
        self.endpoint = "https://initial-resources.cognitiveservices.azure.com/"
        self.deployment = "gpt-4.1"
        self.subscription_key = Open_ai_key
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2024-12-01-preview",
        )

    def _safe_json_parse(self, raw_content: str, fallback_key: str) -> List:
        """Handles malformed JSON, markdown blocks, and unterminated strings."""
        try:
            # Strip markdown if present
            clean = re.sub(r'```json\s*|\s*```', '', raw_content.strip(), flags=re.MULTILINE)
            data = json.loads(clean)
            if isinstance(data, dict):
                # Search for any list in the object if the fallback key is missing
                if fallback_key in data: return data[fallback_key]
                for key in ['resolutions', 'questions', 'results', 'ambiguous_references']:
                    if key in data: return data[key]
                for val in data.values():
                    if isinstance(val, list): return val
            return data if isinstance(data, list) else []
        except Exception:
            # Final fallback: Regex extraction for specific patterns
            if fallback_key == 'questions':
                return re.findall(r'"([^"]+\?)"', raw_content)
            return []

    def evaluate_dataset(self, dataset_path: str, target_user: str = "User A") -> EvaluationResult:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        filename = os.path.basename(dataset_path)
        scenario_type = next((s for s in ['friends_meeting', 'work_collaboration', 'doctor_visit'] if s in filename), 'unknown')
        
        target_utterances = [t for t in dataset['conversation_transcript'] if t['speaker'] == target_user]
        user_key = 'user_a' if target_user == 'User A' else 'user_b'
        target_context = dataset['mobile_context_snapshot'][user_key]
        
        target_gt_res = [gt for gt in dataset['ground_truth_resolutions'] 
                         if any(t['turn_id'] == gt['trigger_turn_id'] and t['speaker'] == target_user for t in dataset['conversation_transcript'])]
        
        print(f"  [1/4] Running Baseline Resolution...")
        bl_res = self._run_baseline_context_aggregation(target_utterances, target_context)
        print(f"  [2/4] Running Framework Resolution...")
        fw_res = self._run_framework_context_aggregation(target_utterances, target_context)
        print(f"  [3/4] Running Baseline Info Gap...")
        bl_queries = self._run_baseline_info_gap(target_utterances, target_context)
        print(f"  [4/4] Running Framework Info Gap...")
        fw_queries = self._run_framework_info_gap(target_utterances, fw_res)

        return EvaluationResult(
            dataset_name=filename, scenario_type=scenario_type, target_user=target_user,
            baseline_resolutions=self._evaluate_resolution_accuracy(bl_res, target_gt_res),
            framework_resolutions=self._evaluate_resolution_accuracy(fw_res, target_gt_res),
            baseline_queries=self._evaluate_query_filtering(bl_queries, dataset['required_protocol_queries']),
            framework_queries=self._evaluate_query_filtering(fw_queries, dataset['required_protocol_queries']),
            baseline_resolutions_raw=bl_res, framework_resolutions_raw=fw_res,
            baseline_queries_raw=bl_queries, framework_queries_raw=fw_queries
        )

    def _run_baseline_context_aggregation(self, utterances, context):
        conversation = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in utterances])
        prompt = f"Resolve ambiguous references using context: {context}. Conversation: {conversation}. Return JSON list."
        response = self.client.chat.completions.create(
            model=self.deployment, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.3
        )
        return self._safe_json_parse(response.choices[0].message.content, 'resolutions')

    def _run_framework_context_aggregation(self, utterances, context):
        conversation = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in utterances])
        prompt = f"""Task: Resolve 'here', 'it', 'them', 'then' using Semantic Location and Calendar.
        LOCATION: {context.get('location_semantic')} | CALENDAR: {context.get('calendar_next')}
        CONVERSATION: {conversation}
        Method: Priority to Semantic Names and Calendar events.
        Format JSON: [ {{"turn_id": <int>, "ambiguous_phrase": "...", "resolved_entity": "...", "resolution_type": "spatial|temporal|person|object"}} ]"""
        
        response = self.client.chat.completions.create(
            model=self.deployment, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.3, max_tokens=2500
        )
        return self._safe_json_parse(response.choices[0].message.content, 'resolutions')

    def _run_baseline_info_gap(self, utterances, context):
        conversation = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in utterances])
        prompt = f"What questions should be asked to help this conversation? {conversation}. Return JSON: {{'questions': []}}"
        response = self.client.chat.completions.create(
            model=self.deployment, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.3
        )
        return self._safe_json_parse(response.choices[0].message.content, 'questions')

    def _run_framework_info_gap(self, utterances, framework_res):
        conversation = "\n".join([f"Turn {t['turn_id']}: {t['speaker']}: {t['text']}" for t in utterances])
        prompt = f"""Identify 'Collaborative Gaps'—missing info preventing task success.
        PRIORITIZE: Entities from the OTHER speaker you cannot resolve (e.g. 'the file', 'that folder').
        CONVERSATION: {conversation}
        RESOLVED LOCALLY: {framework_res}
        Return JSON list of specific questions: {{ "questions": [] }}"""
        
        response = self.client.chat.completions.create(
            model=self.deployment, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.3, max_tokens=1500
        )
        return self._safe_json_parse(response.choices[0].message.content, 'questions')

    def _evaluate_resolution_accuracy(self, predicted, ground_truth) -> Dict[str, ResolutionMetrics]:
        results = {}
        for cat in ['spatial', 'temporal', 'person', 'object']:
            gt_items = [g for g in ground_truth if g.get('resolution_type', 'object') == cat]
            correct = 0
            for gt in gt_items:
                for p in predicted:
                    if p.get('turn_id') == gt['trigger_turn_id'] and self._fuzzy_match(p.get('resolved_entity', ''), gt['resolved_entity']):
                        correct += 1
                        break
            results[cat] = ResolutionMetrics(cat, len(gt_items), correct, correct/len(gt_items) if gt_items else 0.0)
        return results

    def _evaluate_query_filtering(self, predicted, gt_queries) -> QueryFilterMetrics:
        high_gt = [q for q in gt_queries if q['query_quality_check'] == 'HIGH_VALUE']
        low_gt = [q for q in gt_queries if q['query_quality_check'] == 'LOW_VALUE']
        tp = sum(1 for p in predicted if self._query_matches_any(p, high_gt))
        fp = sum(1 for p in predicted if self._query_matches_any(p, low_gt))
        fn = len(high_gt) - tp
        tn = len(low_gt) - fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return QueryFilterMetrics(tp, fp, tn, fn, prec, rec, f1, rec, fn/(tp+fn) if (tp+fn)>0 else 0.0)

    def _query_matches_any(self, predicted_query, gt_list):
        if not gt_list: return False
        prompt = f"Does the question '{predicted_query}' semantically match any of these missing info gaps? {[{'gap': g['reason']} for g in gt_list]}. JSON: {{'matches': true/false}}"
        try:
            res = self.client.chat.completions.create(model=self.deployment, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0)
            return json.loads(res.choices[0].message.content).get('matches', False)
        except: return False

    def _fuzzy_match(self, t1, t2):
        t1, t2 = str(t1).lower(), str(t2).lower()
        return t1 in t2 or t2 in t1 or len(set(t1.split()) & set(t2.split())) / max(len(set(t1.split())), 1) > 0.6

    def save_outputs(self, result: EvaluationResult):
        os.makedirs("evaluation/results", exist_ok=True)
        path = f"evaluation/results/{result.dataset_name}_{result.target_user.replace(' ','_')}.json"
        with open(path, 'w') as f:
            json.dump(result.__dict__, f, default=lambda x: x.__dict__, indent=2)
        print(f"  ✓ Results saved to {path}")

    def print_result(self, result: EvaluationResult):
        print(f"\n{'='*30}\nRESULTS: {result.dataset_name} ({result.target_user})\n{'='*30}")
        print(f"Recall (TPR): {result.framework_queries.tpr:.1%}")
        print(f"F1 Score: {result.framework_queries.f1_score:.3f}")
        for cat, m in result.framework_resolutions.items():
            print(f"Accuracy [{cat}]: {m.accuracy:.1%}")

def main():
    evaluator = ProtocolDatasetEvaluator()
    # Replace with your local path for a quick test
    test_file = "path/to/friends_meeting_001.json" 
    if os.path.exists(test_file):
        result = evaluator.evaluate_dataset(test_file, target_user="User A")
        evaluator.print_result(result)
        evaluator.save_outputs(result)

if __name__ == "__main__":
    main()