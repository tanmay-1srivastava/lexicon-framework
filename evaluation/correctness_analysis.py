"""
Show specific examples of what baseline and framework got RIGHT and WRONG
"""
import json
import os

# Load datasets
data_dir = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/data_generation/new_data/generated_datasets"
outputs_dir = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/evaluation/outputs"

def normalize_text(text):
    """Normalize text for comparison"""
    return text.lower().strip().replace("'", "").replace('"', '')

def check_resolution_correctness(predicted, ground_truth):
    """Check if resolution matches ground truth (fuzzy match)"""
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    
    # Exact match
    if pred_norm == gt_norm:
        return True, "exact"
    
    # Partial match (ground truth contained in prediction or vice versa)
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True, "partial"
    
    # Check key entities match
    gt_words = set(gt_norm.split())
    pred_words = set(pred_norm.split())
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'in', 'at', 'to', 'for', 'of', 'on', 'by', 'with'}
    gt_words -= common_words
    pred_words -= common_words
    
    overlap = len(gt_words & pred_words)
    if overlap >= 2 and overlap / max(len(gt_words), 1) >= 0.5:
        return True, "similar"
    
    return False, "wrong"

# Example files
examples = [
    ('doctor_visit_001.json', 'doctor_visit_001_user_a.json'),
    ('friends_meeting_001.json', 'friends_meeting_001_user_a.json'),
    ('work_collaboration_001.json', 'work_collaboration_001_user_a.json')
]

print(f"\n{'='*120}")
print("CORRECTNESS ANALYSIS: What BASELINE and FRAMEWORK got RIGHT vs WRONG")
print(f"{'='*120}\n")

for dataset_file, output_file in examples:
    # Load dataset with ground truth
    with open(os.path.join(data_dir, dataset_file), 'r') as f:
        dataset = json.load(f)
    
    # Load evaluation output
    with open(os.path.join(outputs_dir, output_file), 'r') as f:
        output = json.load(f)
    
    ground_truth_resolutions = {
        (gt['trigger_turn_id'], normalize_text(gt['ambiguous_phrase'])): gt['resolved_entity']
        for gt in dataset['ground_truth_resolutions']
    }
    
    ground_truth_queries = [
        q.get('natural_language_fallback', '').lower().strip() 
        for q in dataset['required_protocol_queries']
        if q.get('natural_language_fallback')
    ]
    
    print(f"\n{'='*120}")
    print(f"SCENARIO: {output['scenario'].upper().replace('_', ' ')}")
    print(f"{'='*120}\n")
    
    # Analyze context aggregation
    print(f"{'─'*120}")
    print("CONTEXT AGGREGATION (Resolution Accuracy)")
    print(f"{'─'*120}\n")
    
    for system, label in [('baseline', 'BASELINE'), ('framework', 'FRAMEWORK')]:
        resolutions = output[system]['resolutions']
        
        correct_examples = []
        wrong_examples = []
        
        for res in resolutions:
            turn_id = res['turn_id']
            phrase = normalize_text(res['ambiguous_phrase'])
            predicted = res['resolved_entity']
            
            # Find ground truth
            gt_key = (turn_id, phrase)
            if gt_key in ground_truth_resolutions:
                gt_entity = ground_truth_resolutions[gt_key]
                is_correct, match_type = check_resolution_correctness(predicted, gt_entity)
                
                example = {
                    'turn': turn_id,
                    'phrase': res['ambiguous_phrase'],
                    'predicted': predicted,
                    'ground_truth': gt_entity,
                    'match_type': match_type
                }
                
                if is_correct:
                    correct_examples.append(example)
                else:
                    wrong_examples.append(example)
        
        print(f"{label} Results: {len(correct_examples)} correct, {len(wrong_examples)} wrong\n")
        
        # Show examples of what they got RIGHT
        if correct_examples:
            print(f"✅ {label} got RIGHT (showing first 2):\n")
            for i, ex in enumerate(correct_examples[:2], 1):
                print(f"  Example {i}:")
                print(f"    Turn {ex['turn']}: '{ex['phrase']}'")
                print(f"    Predicted: {ex['predicted']}")
                print(f"    Ground Truth: {ex['ground_truth']}")
                print(f"    Match: {ex['match_type']}\n")
        
        # Show examples of what they got WRONG
        if wrong_examples:
            print(f"❌ {label} got WRONG (showing first 2):\n")
            for i, ex in enumerate(wrong_examples[:2], 1):
                print(f"  Example {i}:")
                print(f"    Turn {ex['turn']}: '{ex['phrase']}'")
                print(f"    Predicted: {ex['predicted']}")
                print(f"    Ground Truth: {ex['ground_truth']}\n")
        
        print(f"{'-'*120}\n")
    
    # Analyze query filtering
    print(f"{'─'*120}")
    print("INFORMATION GAP DETECTION (Query Filtering)")
    print(f"{'─'*120}\n")
    
    for system, label in [('baseline', 'BASELINE'), ('framework', 'FRAMEWORK')]:
        queries = output[system]['queries']
        
        # Check which queries match ground truth
        matched_queries = []
        unmatched_queries = []
        
        for query in queries:
            query_norm = normalize_text(query)
            
            # Check if it matches any ground truth query
            matched = False
            for gt_query in ground_truth_queries:
                gt_norm = normalize_text(gt_query)
                
                # Fuzzy match
                query_words = set(query_norm.split()) - {'the', 'a', 'an', 'is', 'are', 'do', 'did', 'you', 'have'}
                gt_words = set(gt_norm.split()) - {'the', 'a', 'an', 'is', 'are', 'do', 'did', 'you', 'have'}
                
                if len(query_words & gt_words) >= 2:
                    matched = True
                    matched_queries.append((query, gt_query))
                    break
            
            if not matched:
                unmatched_queries.append(query)
        
        print(f"{label} Results: {len(matched_queries)} matched ground truth, {len(unmatched_queries)} did not match\n")
        
        # Show examples of queries that MATCHED ground truth
        if matched_queries:
            print(f"✅ {label} queries that MATCHED ground truth (showing first 2):\n")
            for i, (query, gt) in enumerate(matched_queries[:2], 1):
                print(f"  Example {i}:")
                print(f"    Generated: {query}")
                print(f"    Ground Truth: {gt}\n")
        
        # Show examples of queries that DIDN'T match
        if unmatched_queries:
            print(f"❌ {label} queries that DIDN'T match ground truth (showing first 2):\n")
            for i, query in enumerate(unmatched_queries[:2], 1):
                print(f"  Example {i}: {query}\n")
        
        print(f"{'-'*120}\n")
    
    # Show what ground truth queries were MISSED by both systems
    all_baseline_queries = [normalize_text(q) for q in output['baseline']['queries']]
    all_framework_queries = [normalize_text(q) for q in output['framework']['queries']]
    
    missed_by_both = []
    for gt_query in ground_truth_queries[:5]:  # Check first 5
        gt_norm = normalize_text(gt_query)
        gt_words = set(gt_norm.split()) - {'the', 'a', 'an', 'is', 'are', 'do', 'did', 'you', 'have'}
        
        baseline_found = any(len(set(bq.split()) & gt_words) >= 2 for bq in all_baseline_queries)
        framework_found = any(len(set(fq.split()) & gt_words) >= 2 for fq in all_framework_queries)
        
        if not baseline_found and not framework_found:
            missed_by_both.append(gt_query)
    
    if missed_by_both:
        print(f"⚠️  Ground truth queries MISSED by BOTH systems:\n")
        for i, query in enumerate(missed_by_both, 1):
            print(f"  {i}. {query}")
        print()

print(f"\n{'='*120}\n")
