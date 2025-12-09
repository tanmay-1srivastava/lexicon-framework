"""
Compact comparison showing 2-3 specific examples per scenario for both baseline and framework
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

# Pick one example from each scenario
example_files = [
    'doctor_visit_001_user_a.json',
    'friends_meeting_001_user_a.json', 
    'work_collaboration_001_user_a.json'
]

print(f"\n{'='*120}")
print("SPECIFIC EXAMPLES: BASELINE vs FRAMEWORK")
print(f"{'='*120}\n")

for example_file in example_files:
    result = next((r for r in results if r['filename'] == example_file), None)
    if not result:
        continue
    
    print(f"\n{'='*120}")
    print(f"SCENARIO: {result['scenario'].upper().replace('_', ' ')}")
    print(f"{'='*120}\n")
    
    baseline_resolutions = result['baseline']['resolutions']
    framework_resolutions = result['framework']['resolutions']
    baseline_queries = result['baseline']['queries']
    framework_queries = result['framework']['queries']
    
    # Show 3 example resolutions from each
    print(f"{'─'*120}")
    print("CONTEXT AGGREGATION - Sample Resolutions")
    print(f"{'─'*120}\n")
    
    print("BASELINE Examples:")
    for i, res in enumerate(baseline_resolutions[:3], 1):
        print(f"\n  Example {i}:")
        print(f"    Turn: {res['turn_id']}")
        print(f"    Ambiguous phrase: '{res['ambiguous_phrase']}'")
        print(f"    Type: {res['resolution_type']}")
        print(f"    Resolution: {res['resolved_entity']}")
    
    print(f"\n{'-'*120}\n")
    
    print("FRAMEWORK Examples:")
    for i, res in enumerate(framework_resolutions[:3], 1):
        print(f"\n  Example {i}:")
        print(f"    Turn: {res['turn_id']}")
        print(f"    Ambiguous phrase: '{res['ambiguous_phrase']}'")
        print(f"    Type: {res['resolution_type']}")
        print(f"    Resolution: {res['resolved_entity']}")
    
    # Show metrics comparison
    baseline_metrics = result['metrics']['resolution_accuracy']['baseline']
    framework_metrics = result['metrics']['resolution_accuracy']['framework']
    
    print(f"\n{'-'*120}")
    print("\nAccuracy Summary:")
    print(f"  BASELINE:  Total resolutions={len(baseline_resolutions)}")
    for cat in ['spatial', 'temporal', 'person', 'object']:
        if cat in baseline_metrics:
            print(f"    {cat.capitalize()}: {baseline_metrics[cat]['correct']}/{baseline_metrics[cat]['total']} correct ({baseline_metrics[cat]['accuracy']*100:.1f}%)")
    
    print(f"\n  FRAMEWORK: Total resolutions={len(framework_resolutions)}")
    for cat in ['spatial', 'temporal', 'person', 'object']:
        if cat in framework_metrics:
            print(f"    {cat.capitalize()}: {framework_metrics[cat]['correct']}/{framework_metrics[cat]['total']} correct ({framework_metrics[cat]['accuracy']*100:.1f}%)")
    
    # Show query examples
    print(f"\n{'─'*120}")
    print("INFORMATION GAP DETECTION - Sample Queries")
    print(f"{'─'*120}\n")
    
    print("BASELINE Examples:")
    for i, q in enumerate(baseline_queries[:3], 1):
        print(f"  {i}. {q}")
    
    print(f"\n{'-'*120}\n")
    
    print("FRAMEWORK Examples:")
    for i, q in enumerate(framework_queries[:3], 1):
        print(f"  {i}. {q}")
    
    # Show metrics comparison
    baseline_query_metrics = result['metrics']['query_filtering']['baseline']
    framework_query_metrics = result['metrics']['query_filtering']['framework']
    
    print(f"\n{'-'*120}")
    print("\nQuery Metrics Summary:")
    print(f"  BASELINE:  Total queries={len(baseline_queries)}, TPR={baseline_query_metrics['tpr']:.1%}, Precision={baseline_query_metrics['precision']:.1%}, F1={baseline_query_metrics['f1_score']:.3f}")
    print(f"  FRAMEWORK: Total queries={len(framework_queries)}, TPR={framework_query_metrics['tpr']:.1%}, Precision={framework_query_metrics['precision']:.1%}, F1={framework_query_metrics['f1_score']:.3f}")
    
    # Key differences
    print(f"\n{'─'*120}")
    print("Key Differences:")
    print(f"{'─'*120}\n")
    
    # Find unique references
    baseline_refs = {(r['turn_id'], r['ambiguous_phrase']) for r in baseline_resolutions}
    framework_refs = {(r['turn_id'], r['ambiguous_phrase']) for r in framework_resolutions}
    
    only_baseline = baseline_refs - framework_refs
    only_framework = framework_refs - baseline_refs
    
    if only_baseline:
        print(f"References ONLY caught by BASELINE (showing first 3):")
        for turn_id, phrase in sorted(only_baseline)[:3]:
            res = next(r for r in baseline_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
            print(f"  • Turn {turn_id}: '{phrase}' ({res['resolution_type']}) → {res['resolved_entity']}")
    
    if only_framework:
        print(f"\nReferences ONLY caught by FRAMEWORK (showing first 3):")
        for turn_id, phrase in sorted(only_framework)[:3]:
            res = next(r for r in framework_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
            print(f"  • Turn {turn_id}: '{phrase}' ({res['resolution_type']}) → {res['resolved_entity']}")
    
    # Show different resolutions for same reference
    common_refs = baseline_refs & framework_refs
    different_resolutions = []
    
    for turn_id, phrase in common_refs:
        b_res = next(r for r in baseline_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
        f_res = next(r for r in framework_resolutions if r['turn_id'] == turn_id and r['ambiguous_phrase'] == phrase)
        
        if b_res['resolved_entity'] != f_res['resolved_entity']:
            different_resolutions.append((turn_id, phrase, b_res['resolved_entity'], f_res['resolved_entity']))
    
    if different_resolutions:
        print(f"\nSame reference, DIFFERENT resolutions (showing first 2):")
        for turn_id, phrase, b_entity, f_entity in different_resolutions[:2]:
            print(f"\n  Turn {turn_id}: '{phrase}'")
            print(f"    BASELINE:  {b_entity}")
            print(f"    FRAMEWORK: {f_entity}")

print(f"\n{'='*120}\n")
