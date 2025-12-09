"""
Batch Evaluation Runner
Runs evaluation on all 9 protocol datasets and aggregates results by scenario
"""

import json
import os
import sys
from typing import List, Dict
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
from protocol_dataset_evaluation import ProtocolDatasetEvaluator, EvaluationResult

class BatchEvaluator:
    """Batch evaluator for all protocol datasets"""
    
    def __init__(self, datasets_dir: str):
        self.datasets_dir = datasets_dir
        self.evaluator = ProtocolDatasetEvaluator()
        self.results: List[EvaluationResult] = []
    
    def run_all_evaluations(self):
        """Run evaluation on all datasets in the directory (dual evaluation for User A and User B)"""
        
        # Find all JSON datasets
        dataset_files = [
            os.path.join(self.datasets_dir, f)
            for f in os.listdir(self.datasets_dir)
            if f.endswith('.json')
        ]
        
        dataset_files.sort()  # Consistent ordering
        
        # Each dataset gets evaluated twice (User A and User B)
        total_evaluations = len(dataset_files) * 2
        
        print(f"\n{'='*80}")
        print(f"BATCH EVALUATION: {len(dataset_files)} datasets × 2 users = {total_evaluations} evaluations")
        print(f"{'='*80}")
        
        eval_count = 0
        for i, dataset_file in enumerate(dataset_files, 1):
            filename = os.path.basename(dataset_file)
            print(f"\n{'='*80}")
            print(f"[{i}/{len(dataset_files)}] DATASET: {filename}")
            print(f"{'='*80}")
            
            try:
                # Evaluate for User A
                eval_count += 1
                print(f"\n[{eval_count}/{total_evaluations}] Evaluating {filename} - User A...")
                result_a = self.evaluator.evaluate_dataset(dataset_file, target_user="User A")
                self.results.append(result_a)
                self.evaluator.print_result(result_a)
                self.evaluator.save_outputs(result_a)
                
                # Evaluate for User B
                eval_count += 1
                print(f"\n[{eval_count}/{total_evaluations}] Evaluating {filename} - User B...")
                result_b = self.evaluator.evaluate_dataset(dataset_file, target_user="User B")
                self.results.append(result_b)
                self.evaluator.print_result(result_b)
                self.evaluator.save_outputs(result_b)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {dataset_file}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"COMPLETED: {len(self.results)}/{len(dataset_files)} datasets evaluated")
        print(f"{'='*80}")
    
    def generate_aggregate_report(self):
        """Generate aggregate statistics by scenario"""
        
        if not self.results:
            print("No results to aggregate!")
            return
        
        # Group results by scenario
        scenarios = {
            'friends_meeting': [],
            'work_collaboration': [],
            'doctor_visit': []
        }
        
        for result in self.results:
            if result.scenario_type in scenarios:
                scenarios[result.scenario_type].append(result)
        
        print(f"\n{'='*80}")
        print("AGGREGATE RESULTS BY SCENARIO")
        print(f"{'='*80}")
        
        # Resolution Accuracy Summary
        print("\n📍 RESOLUTION ACCURACY (Person/Place/Time/Thing)")
        print("━" * 100)
        
        for scenario_name, scenario_results in scenarios.items():
            if not scenario_results:
                continue
            
            print(f"\n{scenario_name.replace('_', ' ').title()} ({len(scenario_results)} datasets)")
            print("─" * 100)
            print(f"{'Category':<15} {'Baseline Avg':<20} {'Framework Avg':<20} {'Improvement'}")
            print("─" * 100)
            
            for category in ['person', 'spatial', 'temporal', 'object']:
                # Calculate averages
                bl_accuracies = [r.baseline_resolutions[category].accuracy for r in scenario_results]
                fw_accuracies = [r.framework_resolutions[category].accuracy for r in scenario_results]
                
                bl_avg = sum(bl_accuracies) / len(bl_accuracies) if bl_accuracies else 0.0
                fw_avg = sum(fw_accuracies) / len(fw_accuracies) if fw_accuracies else 0.0
                improvement = (fw_avg - bl_avg) * 100
                
                category_label = {
                    'person': 'Person (Name)',
                    'spatial': 'Place (Location)',
                    'temporal': 'Time (Temporal)',
                    'object': 'Thing (Object)'
                }[category]
                
                print(f"{category_label:<15} {bl_avg*100:.1f}%{'':<15} {fw_avg*100:.1f}%{'':<15} {improvement:+.1f}%")
        
        # Query Filtering Summary
        print("\n\n🔍 QUERY FILTERING METRICS (TPR/FNR)")
        print("━" * 100)
        
        for scenario_name, scenario_results in scenarios.items():
            if not scenario_results:
                continue
            
            print(f"\n{scenario_name.replace('_', ' ').title()} ({len(scenario_results)} datasets)")
            print("─" * 100)
            print(f"{'Metric':<20} {'Baseline Avg':<20} {'Framework Avg':<20} {'Improvement'}")
            print("─" * 100)
            
            # Calculate averages
            bl_tpr = [r.baseline_queries.tpr for r in scenario_results]
            fw_tpr = [r.framework_queries.tpr for r in scenario_results]
            bl_fnr = [r.baseline_queries.fnr for r in scenario_results]
            fw_fnr = [r.framework_queries.fnr for r in scenario_results]
            bl_precision = [r.baseline_queries.precision for r in scenario_results]
            fw_precision = [r.framework_queries.precision for r in scenario_results]
            bl_f1 = [r.baseline_queries.f1_score for r in scenario_results]
            fw_f1 = [r.framework_queries.f1_score for r in scenario_results]
            
            bl_tpr_avg = sum(bl_tpr) / len(bl_tpr) if bl_tpr else 0.0
            fw_tpr_avg = sum(fw_tpr) / len(fw_tpr) if fw_tpr else 0.0
            bl_fnr_avg = sum(bl_fnr) / len(bl_fnr) if bl_fnr else 0.0
            fw_fnr_avg = sum(fw_fnr) / len(fw_fnr) if fw_fnr else 0.0
            bl_precision_avg = sum(bl_precision) / len(bl_precision) if bl_precision else 0.0
            fw_precision_avg = sum(fw_precision) / len(fw_precision) if fw_precision else 0.0
            bl_f1_avg = sum(bl_f1) / len(bl_f1) if bl_f1 else 0.0
            fw_f1_avg = sum(fw_f1) / len(fw_f1) if fw_f1 else 0.0
            
            print(f"{'TPR (Recall)':<20} {bl_tpr_avg*100:.1f}%{'':<15} {fw_tpr_avg*100:.1f}%{'':<15} {(fw_tpr_avg - bl_tpr_avg)*100:+.1f}%")
            print(f"{'FNR':<20} {bl_fnr_avg*100:.1f}%{'':<15} {fw_fnr_avg*100:.1f}%{'':<15} {(fw_fnr_avg - bl_fnr_avg)*100:+.1f}%")
            print(f"{'Precision':<20} {bl_precision_avg*100:.1f}%{'':<15} {fw_precision_avg*100:.1f}%{'':<15} {(fw_precision_avg - bl_precision_avg)*100:+.1f}%")
            print(f"{'F1 Score':<20} {bl_f1_avg:.3f}{'':<15} {fw_f1_avg:.3f}{'':<15} {(fw_f1_avg - bl_f1_avg):+.3f}")
        
        print("\n" + "━" * 100)
    
    def save_results_to_csv(self, output_dir: str):
        """Save detailed results to CSV files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Resolution Accuracy CSV
        resolution_data = []
        for result in self.results:
            for category in ['person', 'spatial', 'temporal', 'object']:
                bl_metric = result.baseline_resolutions[category]
                fw_metric = result.framework_resolutions[category]
                
                resolution_data.append({
                    'dataset': result.dataset_name,
                    'scenario': result.scenario_type,
                    'category': category,
                    'baseline_correct': bl_metric.correct,
                    'baseline_total': bl_metric.total,
                    'baseline_accuracy': bl_metric.accuracy,
                    'framework_correct': fw_metric.correct,
                    'framework_total': fw_metric.total,
                    'framework_accuracy': fw_metric.accuracy,
                    'improvement': fw_metric.accuracy - bl_metric.accuracy
                })
        
        df_resolution = pd.DataFrame(resolution_data)
        resolution_csv = os.path.join(output_dir, 'resolution_accuracy_results.csv')
        df_resolution.to_csv(resolution_csv, index=False)
        print(f"\n✓ Saved resolution results to: {resolution_csv}")
        
        # Query Filtering CSV
        query_data = []
        for result in self.results:
            bl_q = result.baseline_queries
            fw_q = result.framework_queries
            
            query_data.append({
                'dataset': result.dataset_name,
                'scenario': result.scenario_type,
                'baseline_tp': bl_q.true_positives,
                'baseline_fp': bl_q.false_positives,
                'baseline_tn': bl_q.true_negatives,
                'baseline_fn': bl_q.false_negatives,
                'baseline_tpr': bl_q.tpr,
                'baseline_fnr': bl_q.fnr,
                'baseline_precision': bl_q.precision,
                'baseline_f1': bl_q.f1_score,
                'framework_tp': fw_q.true_positives,
                'framework_fp': fw_q.false_positives,
                'framework_tn': fw_q.true_negatives,
                'framework_fn': fw_q.false_negatives,
                'framework_tpr': fw_q.tpr,
                'framework_fnr': fw_q.fnr,
                'framework_precision': fw_q.precision,
                'framework_f1': fw_q.f1_score
            })
        
        df_query = pd.DataFrame(query_data)
        query_csv = os.path.join(output_dir, 'query_filtering_results.csv')
        df_query.to_csv(query_csv, index=False)
        print(f"✓ Saved query filtering results to: {query_csv}")
        
        # Summary Report
        report_file = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"PROTOCOL DATASET EVALUATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Datasets Evaluated: {len(self.results)}\n")
            f.write("="*80 + "\n\n")
            
            # Group by scenario
            scenarios = {}
            for result in self.results:
                if result.scenario_type not in scenarios:
                    scenarios[result.scenario_type] = []
                scenarios[result.scenario_type].append(result)
            
            for scenario_name, scenario_results in scenarios.items():
                f.write(f"\n{scenario_name.replace('_', ' ').title()}\n")
                f.write("-"*80 + "\n")
                
                # Resolution accuracy summary
                f.write("\nResolution Accuracy:\n")
                for category in ['person', 'spatial', 'temporal', 'object']:
                    bl_avg = sum(r.baseline_resolutions[category].accuracy for r in scenario_results) / len(scenario_results)
                    fw_avg = sum(r.framework_resolutions[category].accuracy for r in scenario_results) / len(scenario_results)
                    
                    f.write(f"  {category.capitalize()}: Baseline={bl_avg*100:.1f}%, Framework={fw_avg*100:.1f}%, Δ={((fw_avg - bl_avg)*100):+.1f}%\n")
                
                # Query filtering summary
                f.write("\nQuery Filtering:\n")
                bl_tpr_avg = sum(r.baseline_queries.tpr for r in scenario_results) / len(scenario_results)
                fw_tpr_avg = sum(r.framework_queries.tpr for r in scenario_results) / len(scenario_results)
                bl_fnr_avg = sum(r.baseline_queries.fnr for r in scenario_results) / len(scenario_results)
                fw_fnr_avg = sum(r.framework_queries.fnr for r in scenario_results) / len(scenario_results)
                
                f.write(f"  TPR: Baseline={bl_tpr_avg*100:.1f}%, Framework={fw_tpr_avg*100:.1f}%, Δ={((fw_tpr_avg - bl_tpr_avg)*100):+.1f}%\n")
                f.write(f"  FNR: Baseline={bl_fnr_avg*100:.1f}%, Framework={fw_fnr_avg*100:.1f}%, Δ={((fw_fnr_avg - bl_fnr_avg)*100):+.1f}%\n")
                
                f.write("\n")
        
        print(f"✓ Saved summary report to: {report_file}")

def main():
    """Run batch evaluation"""
    
    datasets_dir = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/data_generation/new_data/generated_datasets"
    output_dir = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/evaluation/results"
    
    batch_evaluator = BatchEvaluator(datasets_dir)
    
    # Run all evaluations
    batch_evaluator.run_all_evaluations()
    
    # Generate aggregate report
    batch_evaluator.generate_aggregate_report()
    
    # Save results
    batch_evaluator.save_results_to_csv(output_dir)
    
    print(f"\n{'='*80}")
    print("BATCH EVALUATION COMPLETED")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
