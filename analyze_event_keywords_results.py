"""
Analyze Event Keywords Results - Summary Statistics
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_results():
    """Analyze all result files and generate statistics"""
    
    results_dir = Path("evaluation/event_keywords_results")
    
    # Track statistics
    total_files = 0
    total_gaps_user_a = 0
    total_gaps_user_b = 0
    total_resolutions_user_a = 0
    total_resolutions_user_b = 0
    
    # By scenario type
    scenario_stats = defaultdict(lambda: {
        'files': 0,
        'gaps_a': 0,
        'gaps_b': 0,
        'res_a': 0,
        'res_b': 0
    })
    
    # Priority breakdown
    priority_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0}
    
    # Resolution type breakdown
    resolution_types = {'spatial': 0, 'temporal': 0, 'unknown': 0}
    
    # Process each file
    for file_path in sorted(results_dir.glob("*.json")):
        total_files += 1
        
        # Get scenario type from filename
        scenario = file_path.name.split('_')[0]
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # User A stats
        user_a = data.get('user_a', {})
        gaps_a = user_a.get('info_gaps', [])
        res_a = user_a.get('resolutions', [])
        
        total_gaps_user_a += len(gaps_a)
        total_resolutions_user_a += len(res_a)
        
        scenario_stats[scenario]['gaps_a'] += len(gaps_a)
        scenario_stats[scenario]['res_a'] += len(res_a)
        
        # Priority breakdown
        for gap in gaps_a:
            priority = gap.get('priority', 'MEDIUM')
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        # Resolution types
        for res in res_a:
            res_type = res.get('resolution_type', 'unknown')
            if res_type in resolution_types:
                resolution_types[res_type] += 1
        
        # User B stats
        user_b = data.get('user_b', {})
        gaps_b = user_b.get('info_gaps', [])
        res_b = user_b.get('resolutions', [])
        
        total_gaps_user_b += len(gaps_b)
        total_resolutions_user_b += len(res_b)
        
        scenario_stats[scenario]['gaps_b'] += len(gaps_b)
        scenario_stats[scenario]['res_b'] += len(res_b)
        scenario_stats[scenario]['files'] += 1
        
        # Priority breakdown for User B
        for gap in gaps_b:
            priority = gap.get('priority', 'MEDIUM')
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        # Resolution types for User B
        for res in res_b:
            res_type = res.get('resolution_type', 'unknown')
            if res_type in resolution_types:
                resolution_types[res_type] += 1
    
    # Print results
    print("="*80)
    print("EVENT KEYWORDS DATASET - ANALYSIS RESULTS")
    print("="*80)
    print()
    
    print("📊 OVERALL STATISTICS")
    print("-"*80)
    print(f"Total datasets processed: {total_files}/100")
    print()
    
    print("INFORMATION GAP DETECTION:")
    print(f"  User A gaps detected: {total_gaps_user_a:,}")
    print(f"  User B gaps detected: {total_gaps_user_b:,}")
    print(f"  Total gaps detected: {total_gaps_user_a + total_gaps_user_b:,}")
    print(f"  Average per dataset: {(total_gaps_user_a + total_gaps_user_b)/total_files:.1f}")
    print(f"  Average per user: {(total_gaps_user_a + total_gaps_user_b)/(total_files*2):.1f}")
    print()
    
    print("CONTEXT RESOLUTION (Spatial/Temporal):")
    print(f"  User A resolutions: {total_resolutions_user_a:,}")
    print(f"  User B resolutions: {total_resolutions_user_b:,}")
    print(f"  Total resolutions: {total_resolutions_user_a + total_resolutions_user_b:,}")
    print(f"  Average per dataset: {(total_resolutions_user_a + total_resolutions_user_b)/total_files:.1f}")
    print(f"  Average per user: {(total_resolutions_user_a + total_resolutions_user_b)/(total_files*2):.1f}")
    print()
    
    print("📈 PRIORITY BREAKDOWN (Info Gaps)")
    print("-"*80)
    total_priority = sum(priority_counts.values())
    for priority in ['CRITICAL', 'HIGH', 'MEDIUM']:
        count = priority_counts[priority]
        pct = (count/total_priority*100) if total_priority > 0 else 0
        print(f"  {priority:8s}: {count:5,} ({pct:5.1f}%)")
    print()
    
    print("📍 RESOLUTION TYPE BREAKDOWN")
    print("-"*80)
    total_res = sum(resolution_types.values())
    for res_type in ['spatial', 'temporal', 'unknown']:
        count = resolution_types[res_type]
        pct = (count/total_res*100) if total_res > 0 else 0
        print(f"  {res_type.capitalize():10s}: {count:5,} ({pct:5.1f}%)")
    print()
    
    print("📂 BY SCENARIO TYPE")
    print("-"*80)
    print(f"{'Scenario':<20} {'Files':>6} {'Gaps A':>8} {'Gaps B':>8} {'Res A':>7} {'Res B':>7}")
    print("-"*80)
    
    for scenario in sorted(scenario_stats.keys()):
        stats = scenario_stats[scenario]
        print(f"{scenario:<20} {stats['files']:>6} {stats['gaps_a']:>8} "
              f"{stats['gaps_b']:>8} {stats['res_a']:>7} {stats['res_b']:>7}")
    
    print("="*80)
    
    # Calculate some key metrics
    print()
    print("🔑 KEY METRICS")
    print("-"*80)
    print(f"Info Gap Detection Rate: {(total_gaps_user_a + total_gaps_user_b)/(total_files*2):.2f} gaps/user")
    print(f"Context Resolution Rate: {(total_resolutions_user_a + total_resolutions_user_b)/(total_files*2):.2f} resolutions/user")
    print(f"Critical Gap Percentage: {priority_counts['CRITICAL']/total_priority*100:.1f}%")
    print(f"Spatial vs Temporal: {resolution_types['spatial']}/{resolution_types['temporal']} "
          f"({resolution_types['spatial']/(resolution_types['spatial']+resolution_types['temporal'])*100:.1f}% spatial)")
    print("="*80)

if __name__ == "__main__":
    analyze_results()
