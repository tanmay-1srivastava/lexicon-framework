"""
Detailed comparison showing specific examples of what baseline and framework got right/wrong
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

print(f"\n{'='*120}")
print("DETAILED COMPARISON: BASELINE vs FRAMEWORK")
print(f"{'='*120}\n")

# Pick one example from each scenario to show detailed comparison
example_files = [
    'doctor_visit_001_user_a.json',
    'friends_meeting_001_user_a.json', 
    'work_collaboration_001_user_a.json'
]

for example_file in example_files:
    # Find the result
    result = next((r for r in results if r['filename'] == example_file), None)
    if not result:
        continue
    
    print(f"\n{'='*120}")
    print(f"EXAMPLE: {result['scenario'].upper().replace('_', ' ')} - {result['target_user']}")
    print(f"{'='*120}\n")
    
    # Show all resolutions side by side
    print(f"{'─'*120}")
    print("CONTEXT AGGREGATION - ALL RESOLUTIONS")
    print(f"{'─'*120}\n")
    
    baseline_resolutions = result['baseline']['resolutions']
    framework_resolutions = result['framework']['resolutions']
    
    # Get metrics for this specific result
    baseline_metrics = result['metrics']['resolution_accuracy']['baseline']
    framework_metrics = result['metrics']['resolution_accuracy']['framework']
    
    print(f"BASELINE RESOLUTIONS ({len(baseline_resolutions)} total):")
    print(f"Accuracy by category:")
    for cat in ['spatial', 'temporal', 'person', 'object']:
        if cat in baseline_metrics:
            total = baseline_metrics[cat]['total']
            correct = baseline_metrics[cat]['correct']
            acc = baseline_metrics[cat]['accuracy'] * 100
            print(f"  {cat.capitalize()}: {correct}/{total} correct ({acc:.1f}%)")
    
    print(f"\nAll baseline resolutions:")
    for i, res in enumerate(baseline_resolutions, 1):
        print(f"\n  [{i}] Turn {res['turn_id']}")
        print(f"      Reference: '{res['ambiguous_phrase']}'")
        print(f"      Type: {res['resolution_type']}")
        print(f"      Resolution: {res['resolved_entity']}")
    
    print(f"\n{'─'*120}\n")
    
    print(f"FRAMEWORK RESOLUTIONS ({len(framework_resolutions)} total):")
    print(f"Accuracy by category:")
    for cat in ['spatial', 'temporal', 'person', 'object']:
        if cat in framework_metrics:
            total = framework_metrics[cat]['total']
            correct = framework_metrics[cat]['correct']
            acc = framework_metrics[cat]['accuracy'] * 100
            print(f"  {cat.capitalize()}: {correct}/{total} correct ({acc:.1f}%)")
    
    print(f"\nAll framework resolutions:")
    for i, res in enumerate(framework_resolutions, 1):
        print(f"\n  [{i}] Turn {res['turn_id']}")
        print(f"      Reference: '{res['ambiguous_phrase']}'")
        print(f"      Type: {res['resolution_type']}")
        print(f"      Resolution: {res['resolved_entity']}")
    
    print(f"\n{'─'*120}")
    print("INFORMATION GAP DETECTION - ALL QUERIES")
    print(f"{'─'*120}\n")
    
    baseline_queries = result['baseline']['queries']
    framework_queries = result['framework']['queries']
    
    baseline_query_metrics = result['metrics']['query_filtering']['baseline']
    framework_query_metrics = result['metrics']['query_filtering']['framework']
    
    print(f"BASELINE QUERIES ({len(baseline_queries)} total):")
    print(f"Metrics: TPR={baseline_query_metrics['tpr']:.1%}, Precision={baseline_query_metrics['precision']:.1%}, F1={baseline_query_metrics['f1_score']:.3f}")
    print(f"TP={baseline_query_metrics['tp']}, FP={baseline_query_metrics['fp']}, FN={baseline_query_metrics['fn']}\n")
    for i, q in enumerate(baseline_queries, 1):
        print(f"  [{i}] {q}")
    
    print(f"\n{'─'*120}\n")
    
    print(f"FRAMEWORK QUERIES ({len(framework_queries)} total):")
    print(f"Metrics: TPR={framework_query_metrics['tpr']:.1%}, Precision={framework_query_metrics['precision']:.1%}, F1={framework_query_metrics['f1_score']:.3f}")
    print(f"TP={framework_query_metrics['tp']}, FP={framework_query_metrics['fp']}, FN={framework_query_metrics['fn']}\n")
    for i, q in enumerate(framework_queries, 1):
        print(f"  [{i}] {q}")
    
    print(f"\n{'='*120}")
    print(f"COMPARISON SUMMARY FOR THIS EXAMPLE")
    print(f"{'='*120}\n")
    
    # Compare resolution counts
    baseline_res_count = len(baseline_resolutions)
    framework_res_count = len(framework_resolutions)
    
    print(f"Resolution Count:")
    print(f"  Baseline identified {baseline_res_count} ambiguous references")
    print(f"  Framework identified {framework_res_count} ambiguous references")
    print(f"  Difference: {framework_res_count - baseline_res_count:+d} references")
    
    # Compare query counts
    baseline_query_count = len(baseline_queries)
    framework_query_count = len(framework_queries)
    
    print(f"\nQuery Count:")
    print(f"  Baseline generated {baseline_query_count} queries")
    print(f"  Framework generated {framework_query_count} queries")
    print(f"  Difference: {framework_query_count - baseline_query_count:+d} queries")
    
    # Key differences
    print(f"\nKey Observations:")
    
    # Find references that only one system detected
    baseline_refs = {(r['turn_id'], r['ambiguous_phrase']) for r in baseline_resolutions}
    framework_refs = {(r['turn_id'], r['ambiguous_phrase']) for r in framework_resolutions}
    
    only_baseline = baseline_refs - framework_refs
    only_framework = framework_refs - baseline_refs
    
    if only_baseline:
        print(f"\n  References ONLY detected by Baseline:")
        for turn_id, phrase in sorted(only_baseline):
            res = next(r for r in baseline_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
            print(f"    • Turn {turn_id}: '{phrase}' → {res['resolved_entity']}")
    
    if only_framework:
        print(f"\n  References ONLY detected by Framework:")
        for turn_id, phrase in sorted(only_framework):
            res = next(r for r in framework_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
            print(f"    • Turn {turn_id}: '{phrase}' → {res['resolved_entity']}")
    
    # Compare resolutions for same references
    common_refs = baseline_refs & framework_refs
    if common_refs:
        print(f"\n  Different resolutions for same reference:")
        for turn_id, phrase in sorted(common_refs)[:5]:  # Show first 5
            b_res = next(r for r in baseline_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
            f_res = next(r for r in framework_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
            
            if b_res['resolved_entity'] != f_res['resolved_entity']:
                print(f"    • Turn {turn_id}: '{phrase}'")
                print(f"      Baseline:  {b_res['resolved_entity']}")
                print(f"      Framework: {f_res['resolved_entity']}")

print(f"\n{'='*120}\n")
