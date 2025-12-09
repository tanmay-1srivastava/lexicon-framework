"""
Single Dataset Generator with Validation
Generates a dataset with timestamp in filename and validates quality for:
1. Context Aggregation (spatial, temporal, object resolutions)
2. Info Gap Detection (actionable queries, no small talk)
"""
from generate_dataset_azure import generate_dataset
from datetime import datetime
import json

# Test with a friends meeting scenario
scenario = "Two close friends meeting at a coffee shop to plan a surprise birthday party for their mutual friend while keeping it secret"

# Add timestamp to filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"generated_datasets/test_friends_{timestamp}.json"

print("=" * 70)
print("🎯 GENERATING & VALIDATING TEST DATASET")
print("=" * 70)
print(f"\nScenario: {scenario}")
print(f"Output: {output_file}")
print("\n" + "=" * 70 + "\n")

try:
    dataset = generate_dataset(scenario, output_file)
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS!")
    print("=" * 70)
    print(f"\nDataset saved to: {output_file}")
    print(f"\nQuick Stats:")
    print(f"  - Dataset ID: {dataset.get('dataset_id')}")
    print(f"  - Conversation turns: {len(dataset.get('conversation_transcript', []))}")
    print(f"  - Context resolutions: {len(dataset.get('ground_truth_resolutions', []))}")
    print(f"  - Protocol queries: {len(dataset.get('required_protocol_queries', []))}")
    print(f"  - Relationship: {dataset.get('backstory', {}).get('relationship', 'N/A')}")
    
    # Validate dataset quality for context aggregation and info gap detection
    print("\n" + "=" * 70)
    print("📊 DATASET QUALITY VALIDATION")
    print("=" * 70)
    
    # Context Aggregation Analysis
    print("\n🔍 CONTEXT AGGREGATION COVERAGE:")
    resolutions = dataset.get('ground_truth_resolutions', [])
    
    # Categorize resolution types
    spatial_refs = []  # here, there, this place
    temporal_refs = []  # now, then, later
    object_refs = []  # this, that, it
    person_refs = []  # him, her, them
    
    for res in resolutions:
        phrase = res.get('ambiguous_phrase', '').lower()
        if any(word in phrase for word in ['here', 'there', 'place', 'spot', 'location', 'area', 'building', 'intersection', 'depot', 'gym', 'café']):
            spatial_refs.append(res)
        elif any(word in phrase for word in ['now', 'then', 'later', 'today', 'tomorrow', 'time']):
            temporal_refs.append(res)
        elif any(word in phrase for word in ['this', 'that', 'it', 'box', 'folder']):
            object_refs.append(res)
        elif any(word in phrase for word in ['him', 'her', 'them', 'they', 'he', 'she']):
            person_refs.append(res)
    
    print(f"  ✅ Spatial (Location) Resolutions: {len(spatial_refs)}")
    if len(spatial_refs) > 0:
        print(f"     Examples: {', '.join([r['ambiguous_phrase'] for r in spatial_refs[:3]])}")
    
    print(f"  ✅ Temporal (Time) Resolutions: {len(temporal_refs)}")
    if len(temporal_refs) > 0:
        print(f"     Examples: {', '.join([r['ambiguous_phrase'] for r in temporal_refs[:3]])}")
    
    print(f"  ✅ Object Resolutions: {len(object_refs)}")
    if len(object_refs) > 0:
        print(f"     Examples: {', '.join([r['ambiguous_phrase'] for r in object_refs[:3]])}")
    
    print(f"  ✅ Person Resolutions: {len(person_refs)}")
    if len(person_refs) > 0:
        print(f"     Examples: {', '.join([r['ambiguous_phrase'] for r in person_refs[:3]])}")
    
    print(f"  📊 Total Context Resolutions: {len(resolutions)}")
    
    # Info Gap Detection Analysis
    print("\n🔍 INFO GAP DETECTION QUALITY:")
    queries = dataset.get('required_protocol_queries', [])
    
    # Categorize query types
    actionable_queries = []
    sensitive_queries = []
    
    for query in queries:
        quality = query.get('query_quality_check', '')
        reason = query.get('reason', '').lower()
        intent = query.get('protocol_payload', {}).get('intent', '')
        
        if quality == 'HIGH_VALUE':
            if 'privacy' in intent.lower() or 'permission' in intent.lower():
                sensitive_queries.append(query)
            else:
                actionable_queries.append(query)
    
    print(f"  ✅ Actionable Queries (Location/Time/Contact): {len(actionable_queries)}")
    if len(actionable_queries) > 0:
        print(f"     Example intents: {', '.join([q['protocol_payload']['intent'] for q in actionable_queries[:3]])}")
    
    print(f"  ✅ Sensitive Queries (Privacy/Permissions): {len(sensitive_queries)}")
    if len(sensitive_queries) > 0:
        print(f"     Example intents: {', '.join([q['protocol_payload']['intent'] for q in sensitive_queries[:2]])}")
    
    print(f"  📊 Total High-Value Queries: {len(queries)}")
    
    # Quality checks
    print("\n🎯 QUALITY CHECKS:")
    
    has_spatial = len(spatial_refs) >= 5
    has_temporal = len(temporal_refs) >= 2
    has_object = len(object_refs) >= 2
    has_person = len(person_refs) >= 3
    has_actionable = len(actionable_queries) >= 6
    has_sensitive = len(sensitive_queries) >= 1
    
    print(f"  {'✅' if has_spatial else '⚠️ '} Spatial resolutions (need 5+): {len(spatial_refs)}")
    print(f"  {'✅' if has_temporal else '⚠️ '} Temporal resolutions (need 2+): {len(temporal_refs)}")
    print(f"  {'✅' if has_object else '⚠️ '} Object resolutions (need 2+): {len(object_refs)}")
    print(f"  {'✅' if has_person else '⚠️ '} Person resolutions (need 3+): {len(person_refs)}")
    print(f"  {'✅' if has_actionable else '⚠️ '} Actionable queries (need 6+): {len(actionable_queries)}")
    print(f"  {'✅' if has_sensitive else '⚠️ '} Sensitive queries (need 1+): {len(sensitive_queries)}")
    
    # Check for small talk queries (should be NONE)
    print("\n🚫 SMALL TALK FILTER CHECK:")
    small_talk_found = False
    for query in queries:
        reason = query.get('reason', '').lower()
        if any(word in reason for word in ['joke', 'laugh', 'chat', 'gossip', 'weather', 'greeting', 'hello', 'hi ']):
            small_talk_found = True
            print(f"  ⚠️  WARNING: Potential small talk query detected: {reason[:80]}...")
    
    if not small_talk_found:
        print("  ✅ No small talk queries detected - All queries are actionable/sensitive!")
    
    # Overall assessment
    all_checks_pass = has_spatial and has_temporal and has_object and has_person and has_actionable and has_sensitive and not small_talk_found
    
    print("\n" + "=" * 70)
    if all_checks_pass:
        print("✅ DATASET READY for Context Aggregation & Info Gap Detection!")
        print("\n📁 This dataset is sufficient to test:")
        print("  1. Context Aggregation (context_aggregation/ folder)")
        print("  2. Info Gap Detection (info-gap-detection/ folder)")
    else:
        print("⚠️  DATASET needs improvement in some areas (see above)")
    print("=" * 70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}\n")
    import traceback
    traceback.print_exc()
