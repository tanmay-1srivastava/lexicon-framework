"""
Visualization Module for Evaluation Results
Generates charts for resolution accuracy and query filtering metrics
Works with dual-user evaluations (User A and User B)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
import numpy as np

class ResultVisualizer:
    """Generate visualizations from evaluation results"""
    
    def __init__(self, outputs_dir: str = "evaluation/outputs"):
        self.outputs_dir = outputs_dir
        self.figures_dir = os.path.join("evaluation", "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def load_data_from_json(self):
        """Load data from JSON output files"""
        
        json_files = glob.glob(os.path.join(self.outputs_dir, "*.json"))
        
        resolution_data = []
        query_data = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            dataset = data['dataset'].replace('.json', '')
            scenario = data['scenario']
            target_user = data['target_user']
            
            # Extract resolution accuracy
            for category in ['person', 'spatial', 'temporal', 'object']:
                baseline_res = data['metrics']['resolution_accuracy']['baseline'][category]
                framework_res = data['metrics']['resolution_accuracy']['framework'][category]
                
                resolution_data.append({
                    'dataset': dataset,
                    'scenario': scenario,
                    'target_user': target_user,
                    'category': category,
                    'baseline_total': baseline_res['total'],
                    'baseline_correct': baseline_res['correct'],
                    'baseline_accuracy': baseline_res['accuracy'],
                    'framework_total': framework_res['total'],
                    'framework_correct': framework_res['correct'],
                    'framework_accuracy': framework_res['accuracy']
                })
            
            # Extract query filtering
            baseline_q = data['metrics']['query_filtering']['baseline']
            framework_q = data['metrics']['query_filtering']['framework']
            
            query_data.append({
                'dataset': dataset,
                'scenario': scenario,
                'target_user': target_user,
                'baseline_tpr': baseline_q['tpr'],
                'baseline_fnr': baseline_q['fnr'],
                'baseline_precision': baseline_q['precision'],
                'baseline_f1': baseline_q['f1_score'],
                'framework_tpr': framework_q['tpr'],
                'framework_fnr': framework_q['fnr'],
                'framework_precision': framework_q['precision'],
                'framework_f1': framework_q['f1_score']
            })
        
        self.df_resolution = pd.DataFrame(resolution_data)
        self.df_query = pd.DataFrame(query_data)
        
        print(f"✓ Loaded {len(json_files)} output files")
        print(f"✓ Resolution data: {len(self.df_resolution)} rows")
        print(f"✓ Query filtering data: {len(self.df_query)} rows")
        
        return self.df_resolution, self.df_query
    
    def plot_resolution_accuracy_by_scenario(self):
        """Plot resolution accuracy heatmap by scenario and category"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Resolution Accuracy: Baseline vs Framework by Scenario', fontsize=16, fontweight='bold')
        
        scenarios = ['friends_meeting', 'work_collaboration', 'doctor_visit']
        categories = ['person', 'spatial', 'temporal', 'object']
        
        category_labels = {
            'person': 'Person (Name)',
            'spatial': 'Place (Location)',
            'temporal': 'Time (Temporal)',
            'object': 'Thing (Object)'
        }
        
        for idx, scenario in enumerate(scenarios):
            scenario_data = self.df_resolution[self.df_resolution['scenario'] == scenario]
            
            # Baseline plot
            ax_baseline = axes[0, idx]
            baseline_accuracies = [
                scenario_data[scenario_data['category'] == cat]['baseline_accuracy'].mean() * 100
                for cat in categories
            ]
            
            bars1 = ax_baseline.bar(range(len(categories)), baseline_accuracies, color='#ff7f0e', alpha=0.8)
            ax_baseline.set_ylim(0, 100)
            ax_baseline.set_xticks(range(len(categories)))
            ax_baseline.set_xticklabels([category_labels[c] for c in categories], rotation=45, ha='right')
            ax_baseline.set_ylabel('Accuracy (%)')
            ax_baseline.set_title(f'{scenario.replace("_", " ").title()} - Baseline', fontweight='bold')
            ax_baseline.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax_baseline.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            # Framework plot
            ax_framework = axes[1, idx]
            framework_accuracies = [
                scenario_data[scenario_data['category'] == cat]['framework_accuracy'].mean() * 100
                for cat in categories
            ]
            
            bars2 = ax_framework.bar(range(len(categories)), framework_accuracies, color='#2ca02c', alpha=0.8)
            ax_framework.set_ylim(0, 100)
            ax_framework.set_xticks(range(len(categories)))
            ax_framework.set_xticklabels([category_labels[c] for c in categories], rotation=45, ha='right')
            ax_framework.set_ylabel('Accuracy (%)')
            ax_framework.set_title(f'{scenario.replace("_", " ").title()} - Framework', fontweight='bold')
            ax_framework.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax_framework.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_file = os.path.join(self.figures_dir, 'resolution_accuracy_by_scenario.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_resolution_comparison(self):
        """Plot baseline vs framework comparison for resolution accuracy"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        categories = ['person', 'spatial', 'temporal', 'object']
        category_labels = {
            'person': 'Person\n(Name)',
            'spatial': 'Place\n(Location)',
            'temporal': 'Time\n(Temporal)',
            'object': 'Thing\n(Object)'
        }
        
        scenarios = ['friends_meeting', 'work_collaboration', 'doctor_visit']
        x = np.arange(len(categories))
        width = 0.12
        
        colors_baseline = ['#ff9896', '#ffbb78', '#ffd1a3']
        colors_framework = ['#98df8a', '#2ca02c', '#1a7a1a']
        
        for i, scenario in enumerate(scenarios):
            scenario_data = self.df_resolution[self.df_resolution['scenario'] == scenario]
            
            baseline_accs = [
                scenario_data[scenario_data['category'] == cat]['baseline_accuracy'].mean() * 100
                for cat in categories
            ]
            framework_accs = [
                scenario_data[scenario_data['category'] == cat]['framework_accuracy'].mean() * 100
                for cat in categories
            ]
            
            # Plot baseline and framework side by side
            offset_bl = (i * 2 - 2) * width
            offset_fw = (i * 2 - 1) * width
            
            ax.bar(x + offset_bl, baseline_accs, width, label=f'{scenario.replace("_", " ").title()} - Baseline',
                   color=colors_baseline[i], alpha=0.8)
            ax.bar(x + offset_fw, framework_accs, width, label=f'{scenario.replace("_", " ").title()} - Framework',
                   color=colors_framework[i], alpha=0.8)
        
        ax.set_xlabel('Resolution Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Resolution Accuracy: Baseline vs Framework Across All Scenarios', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([category_labels[c] for c in categories])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.figures_dir, 'resolution_comparison_grouped.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_query_filtering_metrics(self):
        """Plot TPR and FNR comparison for query filtering"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Query Filtering Metrics: Baseline vs Framework', fontsize=16, fontweight='bold')
        
        scenarios = ['friends_meeting', 'work_collaboration', 'doctor_visit']
        metrics = ['tpr', 'fnr', 'precision']
        metric_labels = {
            'tpr': 'True Positive Rate (Recall)',
            'fnr': 'False Negative Rate',
            'precision': 'Precision'
        }
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            scenario_data = self.df_query[self.df_query['scenario'] == scenario]
            
            # Calculate averages
            baseline_tpr = scenario_data['baseline_tpr'].mean() * 100
            framework_tpr = scenario_data['framework_tpr'].mean() * 100
            baseline_fnr = scenario_data['baseline_fnr'].mean() * 100
            framework_fnr = scenario_data['framework_fnr'].mean() * 100
            baseline_precision = scenario_data['baseline_precision'].mean() * 100
            framework_precision = scenario_data['framework_precision'].mean() * 100
            
            x = np.arange(3)
            width = 0.35
            
            baseline_values = [baseline_tpr, baseline_fnr, baseline_precision]
            framework_values = [framework_tpr, framework_fnr, framework_precision]
            
            bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#ff7f0e', alpha=0.8)
            bars2 = ax.bar(x + width/2, framework_values, width, label='Framework', color='#2ca02c', alpha=0.8)
            
            ax.set_ylabel('Percentage (%)', fontsize=11)
            ax.set_title(scenario.replace('_', ' ').title(), fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['TPR', 'FNR', 'Precision'])
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_file = os.path.join(self.figures_dir, 'query_filtering_metrics.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_f1_score_comparison(self):
        """Plot F1 score comparison across scenarios"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenarios = ['friends_meeting', 'work_collaboration', 'doctor_visit']
        scenario_labels = [s.replace('_', ' ').title() for s in scenarios]
        
        baseline_f1 = []
        framework_f1 = []
        
        for scenario in scenarios:
            scenario_data = self.df_query[self.df_query['scenario'] == scenario]
            baseline_f1.append(scenario_data['baseline_f1'].mean())
            framework_f1.append(scenario_data['framework_f1'].mean())
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline', color='#ff7f0e', alpha=0.8)
        bars2 = ax.bar(x + width/2, framework_f1, width, label='Framework', color='#2ca02c', alpha=0.8)
        
        ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Query Filtering F1 Score: Baseline vs Framework', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_file = os.path.join(self.figures_dir, 'f1_score_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_improvement_heatmap(self):
        """Plot improvement heatmap (Framework - Baseline)"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Framework Improvement Over Baseline', fontsize=16, fontweight='bold')
        
        # Resolution Accuracy Improvement
        scenarios = ['friends_meeting', 'work_collaboration', 'doctor_visit']
        categories = ['person', 'spatial', 'temporal', 'object']
        
        improvement_matrix = []
        for scenario in scenarios:
            row = []
            for category in categories:
                scenario_data = self.df_resolution[
                    (self.df_resolution['scenario'] == scenario) &
                    (self.df_resolution['category'] == category)
                ]
                improvement = scenario_data['improvement'].mean() * 100
                row.append(improvement)
            improvement_matrix.append(row)
        
        sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=['Person', 'Spatial', 'Temporal', 'Object'],
                   yticklabels=[s.replace('_', ' ').title() for s in scenarios],
                   ax=ax1, cbar_kws={'label': 'Improvement (%)'})
        ax1.set_title('Resolution Accuracy Improvement', fontweight='bold')
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Scenario')
        
        # Query Filtering Improvement
        query_improvement = []
        for scenario in scenarios:
            scenario_data = self.df_query[self.df_query['scenario'] == scenario]
            tpr_improvement = (scenario_data['framework_tpr'].mean() - scenario_data['baseline_tpr'].mean()) * 100
            fnr_improvement = (scenario_data['framework_fnr'].mean() - scenario_data['baseline_fnr'].mean()) * 100
            precision_improvement = (scenario_data['framework_precision'].mean() - scenario_data['baseline_precision'].mean()) * 100
            f1_improvement = (scenario_data['framework_f1'].mean() - scenario_data['baseline_f1'].mean())
            
            query_improvement.append([tpr_improvement, fnr_improvement, precision_improvement, f1_improvement * 100])
        
        sns.heatmap(query_improvement, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=['TPR', 'FNR', 'Precision', 'F1'],
                   yticklabels=[s.replace('_', ' ').title() for s in scenarios],
                   ax=ax2, cbar_kws={'label': 'Improvement (%)'})
        ax2.set_title('Query Filtering Improvement', fontweight='bold')
        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Scenario')
        
        plt.tight_layout()
        output_file = os.path.join(self.figures_dir, 'improvement_heatmap.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        print("[1/6] Loading data from JSON outputs...")
        self.load_data_from_json()
        
        print("\n[2/6] Plotting resolution accuracy by scenario...")
        self.plot_resolution_accuracy_by_scenario()
        
        print("\n[3/6] Plotting resolution comparison...")
        self.plot_resolution_comparison()
        
        print("\n[4/6] Plotting query filtering metrics...")
        self.plot_query_filtering_metrics()
        
        print("\n[5/6] Plotting F1 score comparison...")
        self.plot_f1_score_comparison()
        
        print("\n[6/6] Plotting improvement heatmap...")
        self.plot_improvement_heatmap()
        
        print("\n" + "="*80)
        print(f"ALL VISUALIZATIONS SAVED TO: {self.figures_dir}")
        print("="*80)

def main():
    """Generate all plots from JSON outputs"""
    
    outputs_dir = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/evaluation/outputs"
    
    visualizer = ResultVisualizer(outputs_dir)
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()
