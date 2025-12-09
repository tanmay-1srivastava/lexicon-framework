"""
Analyze baseline vs framework performance - simplified version
"""
import json
import os

outputs_dir = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/evaluation/outputs"

# Load all results
results = []
for filename in sorted(os.listdir(outputs_dir)):
    if filename.endswith('.json'):
        with open(os.path.join(outputs_dir, filename), 'r') as f:
            data = json.load(f)
            data['filename'] = filename
            results.append(data)

print(f"Loaded {len(results)} evaluation results\n")

# Group by scenario
scenarios = {'doctor_visit': [], 'friends_meeting': [], 'work_collaboration': []}
for r in results:
    scenarios[r['scenario']].append(r)

print("="*100)
print("SCENARIO-WISE ANALYSIS")
print("="*100)

for scenario, scenario_results in scenarios.items():
    print(f"\n{'='*100}")
    print(f"SCENARIO: {scenario.upper().replace('_', ' ')}")
    print(f"{'='*100}")
    
    # Aggregate metrics
    baseline_total = baseline_correct = 0
    framework_total = framework_correct = 0
    
    for res in scenario_results:
        baseline_res = res['metrics']['resolution_accuracy']['baseline']
        framework_res = res['metrics']['resolution_accuracy']['framework']
        
        for category in ['spatial', 'temporal', 'person', 'object']:
            if category in baseline_res:
                baseline_total += baseline_res[category]['total']
                baseline_correct += baseline_res[category]['correct']
            
            if category in framework_res:
                framework_total += framework_res[category]['total']
                framework_correct += framework_res[category]['correct']
    
    baseline_acc = (baseline_correct / baseline_total * 100) if baseline_total > 0 else 0
    framework_acc = (framework_correct / framework_total * 100) if framework_total > 0 else 0
    
    print(f"\n📊 CONTEXT AGGREGATION:")
    print(f"  Baseline:  {baseline_correct}/{baseline_total} correct ({baseline_acc:.1f}%)")
    print(f"  Framework: {framework_correct}/{framework_total} correct ({framework_acc:.1f}%)")
    print(f"  Improvement: {framework_acc - baseline_acc:+.1f}%")
    
    # Sample resolutions
    print(f"\n  📋 BASELINE SAMPLE RESOLUTIONS:")
    for i, res in enumerate(scenario_results[0]['baseline']['resolutions'][:3], 1):
        print(f"    {i}. Turn {res['turn_id']}: '{res['ambiguous_phrase']}' → '{res['resolved_entity']}' ({res['resolution_type']})")
    
    print(f"\n  📋 FRAMEWORK SAMPLE RESOLUTIONS:")
    for i, res in enumerate(scenario_results[0]['framework']['resolutions'][:3], 1):
        print(f"    {i}. Turn {res['turn_id']}: '{res['ambiguous_phrase']}' → '{res['resolved_entity']}' ({res['resolution_type']})")
    
    # Info gap metrics (averaged across all results in scenario)
    avg_baseline_f1 = sum(r['metrics']['query_filtering']['baseline']['f1_score'] for r in scenario_results) / len(scenario_results)
    avg_framework_f1 = sum(r['metrics']['query_filtering']['framework']['f1_score'] for r in scenario_results) / len(scenario_results)
    avg_baseline_tpr = sum(r['metrics']['query_filtering']['baseline']['tpr'] for r in scenario_results) / len(scenario_results)
    avg_framework_tpr = sum(r['metrics']['query_filtering']['framework']['tpr'] for r in scenario_results) / len(scenario_results)
    avg_baseline_prec = sum(r['metrics']['query_filtering']['baseline']['precision'] for r in scenario_results) / len(scenario_results)
    avg_framework_prec = sum(r['metrics']['query_filtering']['framework']['precision'] for r in scenario_results) / len(scenario_results)
    
    print(f"\n📊 INFORMATION GAP DETECTION:")
    print(f"  Baseline:  TPR={avg_baseline_tpr:.1%}, Precision={avg_baseline_prec:.1%}, F1={avg_baseline_f1:.3f}")
    print(f"  Framework: TPR={avg_framework_tpr:.1%}, Precision={avg_framework_prec:.1%}, F1={avg_framework_f1:.3f}")
    print(f"  F1 Improvement: {avg_framework_f1 - avg_baseline_f1:+.3f}")
    
    # Sample queries
    print(f"\n  📋 BASELINE QUERIES (sample):")
    for i, q in enumerate(scenario_results[0]['baseline']['queries'][:3], 1):
        print(f"    {i}. '{q}'")
    
    print(f"\n  📋 FRAMEWORK QUERIES (sample):")
    for i, q in enumerate(scenario_results[0]['framework']['queries'][:3], 1):
        print(f"    {i}. '{q}'")

print(f"\n{'='*100}")
print("CATEGORY-WISE ANALYSIS")
print(f"{'='*100}")

for category in ['person', 'spatial', 'temporal', 'object']:
    baseline_total = baseline_correct = 0
    framework_total = framework_correct = 0
    
    for res in results:
        baseline_res = res['metrics']['resolution_accuracy']['baseline']
        framework_res = res['metrics']['resolution_accuracy']['framework']
        
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

print(f"\n{'='*100}")
print("OVERALL SUMMARY")
print(f"{'='*100}")

total_baseline_correct = total_baseline = 0
total_framework_correct = total_framework = 0

for res in results:
    baseline_res = res['metrics']['resolution_accuracy']['baseline']
    framework_res = res['metrics']['resolution_accuracy']['framework']
    
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

avg_baseline_f1 = sum(r['metrics']['query_filtering']['baseline']['f1_score'] for r in results) / len(results)
avg_framework_f1 = sum(r['metrics']['query_filtering']['framework']['f1_score'] for r in results) / len(results)

print(f"\n📊 INFORMATION GAP DETECTION (ALL SCENARIOS):")
print(f"  Baseline F1:  {avg_baseline_f1:.3f}")
print(f"  Framework F1: {avg_framework_f1:.3f}")
print(f"  Improvement: {avg_framework_f1 - avg_baseline_f1:+.3f}")

print(f"\n{'='*100}\n")
