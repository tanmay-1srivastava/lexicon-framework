import os, json, re
from general_processor import GeneralContextProcessor

def clean(t): return re.sub(r'[^\w\s]', '', str(t).lower()).strip()

def fuzzy_match(p, gt):
    p_c, gt_c = clean(p), clean(gt)
    # Match if one contains the other or if word overlap is high
    if p_c in gt_c or gt_c in p_c: return True
    p_words, gt_words = set(p_c.split()), set(gt_c.split())
    if not gt_words: return False
    return len(p_words & gt_words) / len(gt_words) > 0.5 # Lowered to 0.5 for semantic variations

def run_eval():
    proc = GeneralContextProcessor()
    data_dir = "data_generation/new_data/generated_datasets"
    
    print(f"{'Dataset':<25} | {'User':<7} | {'Baseline':<12} | {'Framework (Nuanced)'}")
    print("-" * 85)

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"): continue
        with open(os.path.join(data_dir, filename)) as f:
            data = json.load(f)
        
        ref_time = data.get('generation_metadata', {}).get('generated_at', '2025-12-09')

        for user_key in ['user_a', 'user_b']:
            target_user = "User A" if user_key == 'user_a' else "User B"
            ctx = data['mobile_context_snapshot'][user_key]
            transcript = [t for t in data['conversation_transcript'] if t['speaker'] == target_user]
            
            # NUANCED FILTER: Source must match target user and be present in the target transcript
            local_gt = [g for g in data['ground_truth_resolutions'] 
                        if target_user in g['resolution_source'] and 
                        any(t['turn_id'] == g['trigger_turn_id'] for t in transcript)]
            
            if not local_gt: continue

            bl_preds = proc.resolve_baseline(transcript, ctx)
            fw_preds = proc.resolve_framework(transcript, ctx, ref_time)

            def score(preds, gt_list):
                correct = 0
                for gt in gt_list:
                    gt_p_clean = clean(gt['ambiguous_phrase'])
                    for p in preds:
                        p_p_clean = clean(p.get('ambiguous_phrase', ''))
                        # Match Turn ID and bidirectional phrase overlap
                        if str(p.get('turn_id')) == str(gt['trigger_turn_id']) and \
                           (gt_p_clean in p_p_clean or p_p_clean in gt_p_clean):
                            if fuzzy_match(p.get('resolved_entity', ''), gt['resolved_entity']):
                                correct += 1
                                break
                return correct / len(gt_list)

            bl_acc = score(bl_preds, local_gt)
            fw_acc = score(fw_preds, local_gt)
            print(f"{filename[:25]:<25} | {target_user:<7} | {bl_acc:<12.1%} | {fw_acc:<12.1%}")

if __name__ == "__main__":
    run_eval()