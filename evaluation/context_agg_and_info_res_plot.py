"""
Comprehensive Plotting Code for Lexicon Framework vs Baseline Evaluation
Visualizes the 4 key metrics: Reference Resolution, Completeness, Question Relevance, Question Count
Author: Tanmay Srivastava
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set the style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_plot_results(csv_file_path):
    """Load results and create comprehensive visualizations"""
    
    # Load the data
    df = pd.read_csv(csv_file_path)
    print(f"📊 Loaded {len(df)} evaluation results")
    
    # Create the plots
    create_main_comparison_plot(df)
    create_detailed_analysis_plots(df)
    create_performance_distribution_plots(df)
    create_correlation_analysis(df)
    
    plt.show()
    print("🎨 All plots generated successfully!")

def create_main_comparison_plot(df):
    """Create main comparison plot showing Framework vs Baseline across 4 metrics"""
    
    # Calculate mean scores for each metric
    framework_means = {
        'Reference Resolution': df['framework_reference_resolution'].mean(),
        'Completeness Score': df['framework_completeness_score'].mean(), 
        'Question Relevance': df['framework_question_relevance'].mean(),
        'Avg Question Count': df['framework_question_count'].mean()
    }
    
    baseline_means = {
        'Reference Resolution': df['baseline_reference_resolution'].mean(),
        'Completeness Score': df['baseline_completeness_score'].mean(),
        'Question Relevance': df['baseline_question_relevance'].mean(), 
        'Avg Question Count': df['baseline_question_count'].mean()
    }
    
    # Create the main comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🏆 Lexicon Framework vs Baseline GPT - Performance Comparison', fontsize=16, fontweight='bold')
    
    # Metric 1: Reference Resolution
    metrics_1 = ['Framework', 'Baseline']
    values_1 = [framework_means['Reference Resolution'], baseline_means['Reference Resolution']]
    colors_1 = ['#2E86AB', '#A23B72']
    bars_1 = ax1.bar(metrics_1, values_1, color=colors_1, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('📍 Reference Resolution Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy Score (0-1)')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars_1, values_1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 2: Completeness Score
    values_2 = [framework_means['Completeness Score'], baseline_means['Completeness Score']]
    bars_2 = ax2.bar(metrics_1, values_2, color=colors_1, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('✅ Context Completeness Score', fontweight='bold')
    ax2.set_ylabel('Completeness Score (0-1)')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars_2, values_2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 3: Question Relevance
    values_3 = [framework_means['Question Relevance'], baseline_means['Question Relevance']]
    bars_3 = ax3.bar(metrics_1, values_3, color=colors_1, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('🎯 Question Relevance Score', fontweight='bold')
    ax3.set_ylabel('Relevance Score (0-1)')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars_3, values_3):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 4: Question Count
    values_4 = [framework_means['Avg Question Count'], baseline_means['Avg Question Count']]
    bars_4 = ax4.bar(metrics_1, values_4, color=colors_1, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('📊 Average Question Count', fontweight='bold')
    ax4.set_ylabel('Number of Questions')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars_4, values_4):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lexicon_main_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: lexicon_main_comparison.png")

def create_detailed_analysis_plots(df):
    """Create detailed analysis plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔍 Detailed Performance Analysis - Framework vs Baseline', fontsize=16, fontweight='bold')
    
    # Plot 1: Box plots for all 4 metrics
    metrics_data = []
    labels = []
    
    for metric in ['reference_resolution', 'completeness_score', 'question_relevance']:
        metrics_data.extend([df[f'framework_{metric}'].values, df[f'baseline_{metric}'].values])
        labels.extend([f'Framework\n{metric.replace("_", " ").title()}', f'Baseline\n{metric.replace("_", " ").title()}'])
    
    box_plot = ax1.boxplot(metrics_data, labels=labels, patch_artist=True)
    ax1.set_title('📦 Score Distribution Comparison', fontweight='bold')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Color the boxes
    colors = ['#2E86AB', '#A23B72'] * 3
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot 2: Scatter plot - Framework vs Baseline Performance
    ax2.scatter(df['baseline_question_relevance'], df['framework_question_relevance'], 
               alpha=0.6, s=60, color='#F18F01', edgecolor='black', linewidth=0.5)
    ax2.plot([0, 1], [0, 1], '--', color='red', alpha=0.8, linewidth=2, label='Perfect Agreement')
    ax2.set_xlabel('Baseline Question Relevance')
    ax2.set_ylabel('Framework Question Relevance')
    ax2.set_title('🎯 Question Relevance: Framework vs Baseline', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Efficiency Analysis
    efficiency_ratios = df['efficiency_ratio'].values
    ax3.hist(efficiency_ratios, bins=15, color='#C73E1D', alpha=0.7, edgecolor='black', linewidth=1)
    ax3.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Equal Efficiency')
    ax3.axvline(x=efficiency_ratios.mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean Ratio: {efficiency_ratios.mean():.2f}')
    ax3.set_xlabel('Efficiency Ratio (Framework/Baseline Question Count)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('⚡ Question Efficiency Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Performance by Conversation Type
    conversation_types = df['conversation_format'].value_counts()
    if len(conversation_types) > 1:
        # Group by conversation format and calculate means
        format_performance = df.groupby('conversation_format').agg({
            'framework_completeness_score': 'mean',
            'baseline_completeness_score': 'mean',
            'framework_question_relevance': 'mean',
            'baseline_question_relevance': 'mean'
        }).reset_index()
        
        x = np.arange(len(format_performance))
        width = 0.35
        
        ax4.bar(x - width/2, format_performance['framework_completeness_score'], 
               width, label='Framework Completeness', color='#2E86AB', alpha=0.8)
        ax4.bar(x + width/2, format_performance['baseline_completeness_score'], 
               width, label='Baseline Completeness', color='#A23B72', alpha=0.8)
        
        ax4.set_xlabel('Conversation Type')
        ax4.set_ylabel('Completeness Score')
        ax4.set_title('🗂️ Performance by Conversation Type', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(format_performance['conversation_format'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Single Conversation Type\nNo Comparison Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('🗂️ Performance by Conversation Type', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lexicon_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: lexicon_detailed_analysis.png")

def create_performance_distribution_plots(df):
    """Create performance distribution plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📈 Performance Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot distributions for each metric
    metrics = [
        ('framework_reference_resolution', 'baseline_reference_resolution', 'Reference Resolution'),
        ('framework_completeness_score', 'baseline_completeness_score', 'Completeness Score'),
        ('framework_question_relevance', 'baseline_question_relevance', 'Question Relevance'),
        ('framework_question_count', 'baseline_question_count', 'Question Count')
    ]
    
    for idx, (fw_metric, bl_metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Create histograms
        ax.hist(df[fw_metric], bins=12, alpha=0.7, label='Framework', color='#2E86AB', edgecolor='black')
        ax.hist(df[bl_metric], bins=12, alpha=0.7, label='Baseline', color='#A23B72', edgecolor='black')
        
        # Add mean lines
        fw_mean = df[fw_metric].mean()
        bl_mean = df[bl_metric].mean()
        ax.axvline(fw_mean, color='#2E86AB', linestyle='--', linewidth=2, 
                  label=f'Framework Mean: {fw_mean:.3f}')
        ax.axvline(bl_mean, color='#A23B72', linestyle='--', linewidth=2, 
                  label=f'Baseline Mean: {bl_mean:.3f}')
        
        ax.set_title(f'{title} Distribution', fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lexicon_distributions.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: lexicon_distributions.png")

def create_correlation_analysis(df):
    """Create correlation analysis plots"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('🔗 Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Correlation heatmap for Framework metrics
    framework_metrics = df[['framework_reference_resolution', 'framework_completeness_score', 
                           'framework_question_relevance', 'framework_question_count']].copy()
    framework_metrics.columns = ['Reference Resolution', 'Completeness', 'Question Relevance', 'Question Count']
    
    correlation_fw = framework_metrics.corr()
    sns.heatmap(correlation_fw, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax1, cbar_kws={'label': 'Correlation Coefficient'})
    ax1.set_title('Framework Metrics Correlation', fontweight='bold')
    
    # Correlation heatmap for Baseline metrics
    baseline_metrics = df[['baseline_reference_resolution', 'baseline_completeness_score', 
                          'baseline_question_relevance', 'baseline_question_count']].copy()
    baseline_metrics.columns = ['Reference Resolution', 'Completeness', 'Question Relevance', 'Question Count']
    
    correlation_bl = baseline_metrics.corr()
    sns.heatmap(correlation_bl, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax2, cbar_kws={'label': 'Correlation Coefficient'})
    ax2.set_title('Baseline Metrics Correlation', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lexicon_correlations.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: lexicon_correlations.png")

def print_summary_statistics(df):
    """Print comprehensive summary statistics"""
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE EVALUATION SUMMARY")
    print("="*60)
    
    # Framework vs Baseline comparison
    metrics = [
        ('reference_resolution', 'Reference Resolution Accuracy'),
        ('completeness_score', 'Context Completeness Score'),
        ('question_relevance', 'Question Relevance Score'),
        ('question_count', 'Average Question Count')
    ]
    
    print("\n🏆 FRAMEWORK vs BASELINE COMPARISON:")
    print("-" * 50)
    
    for metric, title in metrics:
        fw_mean = df[f'framework_{metric}'].mean()
        bl_mean = df[f'baseline_{metric}'].mean()
        fw_std = df[f'framework_{metric}'].std()
        bl_std = df[f'baseline_{metric}'].std()
        
        # Statistical significance test
        if metric != 'question_count':
            t_stat, p_value = stats.ttest_rel(df[f'framework_{metric}'], df[f'baseline_{metric}'])
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        else:
            significance = ""
        
        print(f"\n{title}:")
        print(f"  Framework: {fw_mean:.3f} ± {fw_std:.3f}")
        print(f"  Baseline:  {bl_mean:.3f} ± {bl_std:.3f}")
        print(f"  Difference: {fw_mean - bl_mean:+.3f} {significance}")
    
    print(f"\n📈 EFFICIENCY ANALYSIS:")
    print(f"  Average Efficiency Ratio: {df['efficiency_ratio'].mean():.2f}")
    print(f"  Framework generates {df['framework_question_count'].mean():.1f} questions on average")
    print(f"  Baseline generates {df['baseline_question_count'].mean():.1f} questions on average")
    
    # Win/Loss analysis
    fw_wins = 0
    total_comparisons = 0
    
    for metric, _ in metrics[:3]:  # Exclude question_count
        fw_better = (df[f'framework_{metric}'] > df[f'baseline_{metric}']).sum()
        total = len(df)
        fw_wins += fw_better
        total_comparisons += total
        print(f"  Framework outperforms Baseline in {metric}: {fw_better}/{total} files ({fw_better/total*100:.1f}%)")
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"  Framework wins: {fw_wins}/{total_comparisons} comparisons ({fw_wins/total_comparisons*100:.1f}%)")
    
    print("\n" + "="*60)

def main():
    """Main function to run all plotting analysis"""
    
    # File path - update this to match your CSV file location
    csv_file = "evaluation/lexicon_ultra_robust_evaluation.csv"  # Update this path
    
    try:
        # Load and plot results
        df = pd.read_csv(csv_file)
        print(f"📂 Successfully loaded {csv_file}")
        print(f"📊 Processing {len(df)} evaluation results...")
        
        # Create all plots
        load_and_plot_results(csv_file)
        
        # Print summary statistics
        print_summary_statistics(df)
        
        print(f"\n🎨 Generated 4 comprehensive visualization files:")
        print(f"   📊 lexicon_main_comparison.png - Main 4-metric comparison")
        print(f"   📈 lexicon_detailed_analysis.png - Detailed performance analysis") 
        print(f"   📉 lexicon_distributions.png - Performance distributions")
        print(f"   🔗 lexicon_correlations.png - Correlation analysis")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_file}")
        print("   Please update the csv_file variable with the correct path to your results file")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()