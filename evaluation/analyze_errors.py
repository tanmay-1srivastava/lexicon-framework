"""
Analyze baseline vs framework performance to understand what was right and wrong
"""
import json
import os
from collections import defaultdict
from typing import Dict, List

class PerformanceAnalyzer:
    def __init__(self, outputs_dir: str):
        self.outputs_dir = outputs_dir
        self.results = []
        
    def load_all_results(self):
        """Load all evaluation results"""
        for filename in sorted(os.listdir(self.outputs_dir)):
            if filename.endswith('.json'):
                filepath = os.path.join(self.outputs_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['filename'] = filename
                    self.results.append(data)
        print(f"Loaded {len(self.results)} evaluation results\n")
    
    def analyze_by_scenario(self):
        """Analyze performance by scenario"""
        
        scenarios = {
            'doctor_visit': [],
            'friends_meeting': [],
            'work_collaboration': []
        }
        
        # Group by scenario
        for result in self.results:
            scenario = result['scenario']
            scenarios[scenario].append(result)
        
        print("="*100)
        print("SCENARIO-WISE ANALYSIS")
        print("="*100)
        
        for scenario, results in scenarios.items():
            print(f"\n{'='*100}")
            print(f"SCENARIO: {scenario.upper().replace('_', ' ')}")
            print(f"{'='*100}")
            
            # Aggregate metrics
            baseline_metrics = {
                'context_agg': {'correct': 0, 'total': 0, 'resolutions': []},
                'info_gap': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'queries': []}
            }
            framework_metrics = {
                'context_agg': {'correct': 0, 'total': 0, 'resolutions': []},
                'info_gap': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'queries': []}
            }
            
            for res in results:
                # Get resolution accuracy from metrics
                baseline_res_accuracy = res['metrics']['resolution_accuracy']['baseline']
                framework_res_accuracy = res['metrics']['resolution_accuracy']['framework']
                
                # Aggregate totals
                for category in ['spatial', 'temporal', 'person', 'object']:
                    if category in baseline_res_accuracy:
                        baseline_metrics['context_agg']['total'] += baseline_res_accuracy[category]['total']
                        baseline_metrics['context_agg']['correct'] += baseline_res_accuracy[category]['correct']
                    
                    if category in framework_res_accuracy:
                        framework_metrics['context_agg']['total'] += framework_res_accuracy[category]['total']
                        framework_metrics['context_agg']['correct'] += framework_res_accuracy[category]['correct']
                
                # Store resolutions for inspection
                baseline_metrics['context_agg']['resolutions'].extend(res['baseline']['resolutions'])
                framework_metrics['context_agg']['resolutions'].extend(res['framework']['resolutions'])
                
                # Store queries for inspection
                baseline_metrics['info_gap']['queries'].extend(res['baseline']['queries'])
                framework_metrics['info_gap']['queries'].extend(res['framework']['queries'])
            
            # Calculate context aggregation accuracy
            baseline_acc = (baseline_metrics['context_agg']['correct'] / baseline_metrics['context_agg']['total'] * 100) if baseline_metrics['context_agg']['total'] > 0 else 0
            framework_acc = (framework_metrics['context_agg']['correct'] / framework_metrics['context_agg']['total'] * 100) if framework_metrics['context_agg']['total'] > 0 else 0
            
            print(f"\n📊 CONTEXT AGGREGATION:")
            print(f"  Baseline:  {baseline_metrics['context_agg']['correct']}/{baseline_metrics['context_agg']['total']} correct ({baseline_acc:.1f}%)")
            print(f"  Framework: {framework_metrics['context_agg']['correct']}/{framework_metrics['context_agg']['total']} correct ({framework_acc:.1f}%)")
            print(f"  Improvement: {framework_acc - baseline_acc:+.1f}%")
            
            # Show sample errors
            print(f"\n  📋 BASELINE SAMPLE RESOLUTIONS:")
            baseline_resolutions = baseline_metrics['context_agg']['resolutions'][:5]
            for i, res in enumerate(baseline_resolutions, 1):
                print(f"    {i}. Turn {res.get('turn_id', '?')}: '{res.get('ambiguous_phrase', 'N/A')}' → '{res.get('resolved_entity', 'N/A')}' ({res.get('resolution_type', 'unknown')})")
            
            print(f"\n  📋 FRAMEWORK SAMPLE RESOLUTIONS:")
            framework_resolutions = framework_metrics['context_agg']['resolutions'][:5]
            for i, res in enumerate(framework_resolutions, 1):
                print(f"    {i}. Turn {res.get('turn_id', '?')}: '{res.get('ambiguous_phrase', 'N/A')}' → '{res.get('resolved_entity', 'N/A')}' ({res.get('resolution_type', 'unknown')})")
            
            # Calculate info gap metrics from first result (they're aggregated across users)
            if results:
                sample = results[0]
                
                baseline_tpr = sample['metrics']['query_filtering']['baseline']['tpr']
                baseline_precision = sample['metrics']['query_filtering']['baseline']['precision']
                baseline_f1 = sample['metrics']['query_filtering']['baseline']['f1_score']
                
                framework_tpr = sample['metrics']['query_filtering']['framework']['tpr']
                framework_precision = sample['metrics']['query_filtering']['framework']['precision']
                framework_f1 = sample['metrics']['query_filtering']['framework']['f1_score']
                
                print(f"\n📊 INFORMATION GAP DETECTION:")
                print(f"  Baseline:  TPR={baseline_tpr:.1%}, Precision={baseline_precision:.1%}, F1={baseline_f1:.3f}")
                print(f"  Framework: TPR={framework_tpr:.1%}, Precision={framework_precision:.1%}, F1={framework_f1:.3f}")
                print(f"  F1 Improvement: {framework_f1 - baseline_f1:+.3f}")
                
                # Show sample query predictions
                print(f"\n  ✅ BASELINE CORRECT QUERIES (sample):")
                baseline_correct = [q for q in baseline_metrics['info_gap']['queries'] if q.get('is_match', False)][:3]
                for i, q in enumerate(baseline_correct, 1):
                    print(f"    {i}. '{q.get('query', 'N/A')}'")
                
                print(f"\n  ❌ BASELINE MISSED QUERIES (sample):")
                baseline_missed = [q for q in baseline_metrics['info_gap']['queries'] if not q.get('is_match', False)][:3]
                for i, q in enumerate(baseline_missed, 1):
                    print(f"    {i}. '{q.get('query', 'N/A')}'")
                
                print(f"\n  ✅ FRAMEWORK CORRECT QUERIES (sample):")
                framework_correct = [q for q in framework_metrics['info_gap']['queries'] if q.get('is_match', False)][:3]
                for i, q in enumerate(framework_correct, 1):
                    print(f"    {i}. '{q.get('query', 'N/A')}'")
                
                print(f"\n  ❌ FRAMEWORK MISSED QUERIES (sample):")
                framework_missed = [q for q in framework_metrics['info_gap']['queries'] if not q.get('is_match', False)][:3]
                for i, q in enumerate(framework_missed, 1):
                    print(f"    {i}. '{q.get('query', 'N/A')}'")
    
    def analyze_by_category(self):
        """Analyze resolution accuracy by reference category"""
        
        categories = ['person', 'spatial', 'temporal', 'object']
        
        print(f"\n{'='*100}")
        print("CATEGORY-WISE RESOLUTION ANALYSIS")
        print(f"{'='*100}")
        
        for category in categories:
            baseline_correct = 0
            baseline_total = 0
            framework_correct = 0
            framework_total = 0
            
            baseline_errors = []
            framework_errors = []
            baseline_successes = []
            framework_successes = []
            
            for result in self.results:
                # Get resolution accuracy from metrics
                baseline_res = result['metrics']['resolution_accuracy']['baseline']
                framework_res = result['metrics']['resolution_accuracy']['framework']
                
                if category in baseline_res:
                    baseline_total += baseline_res[category]['total']
                    baseline_correct += baseline_res[category]['correct']
                
                if category in framework_res:
                    framework_total += framework_res[category]['total']
                    framework_correct += framework_res[category]['correct']
            
            baseline_acc = (baseline_correct / baseline_total * 100) if baseline_total > 0 else 0
            framework_acc = (framework_correct / framework_total * 100) if framework_total > 0 else 0
            
            print(f"\n{'─'*100}")
            print(f"📌 {category.upper()} REFERENCES")
            print(f"{'─'*100}")
            print(f"  Baseline:  {baseline_correct}/{baseline_total} correct ({baseline_acc:.1f}%)")
            print(f"  Framework: {framework_correct}/{framework_total} correct ({framework_acc:.1f}%)")
            print(f"  Improvement: {framework_acc - baseline_acc:+.1f}%")
    
    def generate_summary(self):
        """Generate overall summary"""
        
        print(f"\n{'='*100}")
        print("OVERALL SUMMARY")
        print(f"{'='*100}")
        
        total_baseline_correct = 0
        total_baseline = 0
        total_framework_correct = 0
        total_framework = 0
        
        for result in self.results:
            baseline_res = result['metrics']['resolution_accuracy']['baseline']
            framework_res = result['metrics']['resolution_accuracy']['framework']
            
            for category in ['spatial', 'temporal', 'person', 'object']:
                if category in baseline_res:
                    total_baseline += baseline_res[category]['total']
                    total_baseline_correct += baseline_res[category]['correct']
                
                if category in framework_res:
                    total_framework += framework_res[category]['total']
                    total_framework_correct += framework_res[category]['correct']
        
        baseline_acc = (total_baseline_correct / total_baseline * 100) if total_baseline > 0 else 0
        framework_acc = (total_framework_correct / total_framework * 100) if total_framework > 0 else 0
        
        print(f"\n📊 CONTEXT AGGREGATION (ALL SCENARIOS):")
        print(f"  Baseline:  {total_baseline_correct}/{total_baseline} correct ({baseline_acc:.1f}%)")
        print(f"  Framework: {total_framework_correct}/{total_framework} correct ({framework_acc:.1f}%)")
        print(f"  Improvement: {framework_acc - baseline_acc:+.1f}%")
        
        # Average info gap metrics
        avg_baseline_f1 = sum(r['metrics']['query_filtering']['baseline']['f1_score'] for r in self.results) / len(self.results)
        avg_framework_f1 = sum(r['metrics']['query_filtering']['framework']['f1_score'] for r in self.results) / len(self.results)
        
        print(f"\n📊 INFORMATION GAP DETECTION (ALL SCENARIOS):")
        print(f"  Baseline F1:  {avg_baseline_f1:.3f}")
        print(f"  Framework F1: {avg_framework_f1:.3f}")
        print(f"  Improvement: {avg_framework_f1 - avg_baseline_f1:+.3f}")
        
        print(f"\n{'='*100}\n")

def main():
    outputs_dir = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/evaluation/outputs"
    
    analyzer = PerformanceAnalyzer(outputs_dir)
    analyzer.load_all_results()
    
    # Run analyses
    analyzer.analyze_by_scenario()
    analyzer.analyze_by_category()
    analyzer.generate_summary()

if __name__ == "__main__":
    main()
