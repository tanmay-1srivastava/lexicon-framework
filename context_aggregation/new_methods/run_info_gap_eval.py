import os, json, re
from datetime import datetime
import numpy as np
from info_gap_processor import InfoGapProcessor

class InfoGapEvaluator:
    def __init__(self):
        self.proc = InfoGapProcessor()
        self.output_file = None

    def semantic_match(self, pred, gt):
        """LLM-based semantic matching for accurate evaluation."""
        if not pred or not gt: return False
        prompt = f"""Compare these two information gaps. Do they target the same missing data point?
        Q1 (Predicted): {pred}
        Q2 (Ground Truth): {gt}
        
        Criteria:
        - If Q1 is a confirmation of the entity in Q2, it is a MATCH.
        - If they target the same Turn ID and the same noun/action, it is a MATCH.
        
        Return JSON: {{ "match": true/false, "reason": "..." }}"""
        try:
            res = self.proc.client.chat.completions.create(
                model="gpt-4.1", messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0
            )
            return json.loads(res.choices[0].message.content).get('match', False)
        except: return False

    def run_eval(self, data_dir="data_generation/new_data/generated_datasets"):
        # Create output file for detailed logs in same directory as script
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, f"info_gap_eval_detailed_{timestamp}.txt")
        self.output_file = open(output_path, 'w')
        
        # Print header to console
        print(f"{'Dataset':<25} | {'User':<7} | {'Type':<10} | {'TPR':<8} | {'FNR':<8} | {'Prec'}")
        print("-" * 85)
        
        # Write header to file
        self.output_file.write(f"Information Gap Detection Evaluation - Detailed Results\n")
        self.output_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.output_file.write("="*90 + "\n\n")
        
        # Store results for summary statistics
        baseline_tprs = []
        baseline_fnrs = []
        framework_tprs = []
        framework_fnrs = []

        for filename in sorted(os.listdir(data_dir)):
            if not filename.endswith(".json"): continue
            with open(os.path.join(data_dir, filename)) as f:
                data = json.load(f)

            # ONLY EVALUATE USER A (has consistent/comprehensive ground truth)
            for user_key in ['user_a']:
                target = "User A"
                transcript = [t for t in data['conversation_transcript'] if t['speaker'] == target]
                ctx = data['mobile_context_snapshot'][user_key]
                
                hp_gt = [q for q in data['required_protocol_queries'] 
                         if q['query_quality_check'] == 'HIGH_VALUE' and 
                         any(t['turn_id'] == q['trigger_turn_id'] and t['speaker'] == target for t in transcript)]
                
                if not hp_gt: continue

                known_gt = [g for g in data['ground_truth_resolutions'] if target in g['resolution_source']]
                bl_preds = self.proc.detect_baseline(transcript, ctx)
                fw_preds = self.proc.detect_gaps(transcript, ctx, known_gt)

                # Write detailed output to FILE
                self.output_file.write(f"\n{'='*90}\n")
                self.output_file.write(f"FILE: {filename} | USER: {target}\n")
                self.output_file.write(f"{'='*90}\n")
                self.output_file.write(f"[GROUND TRUTH GAPS]:\n")
                for q in hp_gt: 
                    self.output_file.write(f"  - Turn {q['trigger_turn_id']}: {q['natural_language_fallback']}\n")

                def evaluate_and_log(preds, label):
                    self.output_file.write(f"\n[{label.upper()} PREDICTIONS]:\n")
                    
                    # CORRECT: For each GT, check if ANY prediction matches it
                    matched_gts = set()
                    
                    for p in preds:
                        p_text = p.get('question', '') if isinstance(p, dict) else str(p)
                        matched = False
                        for gt in hp_gt:
                            if self.semantic_match(p_text, gt['natural_language_fallback']):
                                matched_gts.add(gt['trigger_turn_id'])
                                matched = True
                                self.output_file.write(f"  ✓ Turn {p.get('turn_id', '?')}: {p_text} -> [MATCHED GT TURN {gt['trigger_turn_id']}]\n")
                                break
                        if not matched:
                            self.output_file.write(f"  ✗ Turn {p.get('turn_id', '?')}: {p_text}\n")
                    
                    tp = len(matched_gts)  # Number of unique GTs matched
                    tpr = tp / len(hp_gt) if hp_gt else 0.0
                    fnr = 1.0 - tpr
                    prec = tp / len(preds) if preds else 0.0
                    return tpr, fnr, prec

                bl_m = evaluate_and_log(bl_preds, "Baseline")
                fw_m = evaluate_and_log(fw_preds, "Framework")

                # Store metrics for summary stats
                baseline_tprs.append(bl_m[0])
                baseline_fnrs.append(bl_m[1])
                framework_tprs.append(fw_m[0])
                framework_fnrs.append(fw_m[1])

                # Print only summary to CONSOLE
                print(f"{filename[:25]:<25} | {target:<7} | {'Baseline':<10} | {bl_m[0]:<8.1%} | {bl_m[1]:<8.1%} | {bl_m[2]:.1%}")
                print(f"{'':<25} | {'':<7} | {'Framework':<10} | {fw_m[0]:<8.1%} | {fw_m[1]:<8.1%} | {fw_m[2]:.1%}")
                
                # Write summary to file too
                self.output_file.write(f"\n{filename[:25]:<25} | {target:<7} | {'Baseline':<10} | {bl_m[0]:<8.1%} | {bl_m[1]:<8.1%} | {bl_m[2]:.1%}\n")
                self.output_file.write(f"{'':<25} | {'':<7} | {'Framework':<10} | {fw_m[0]:<8.1%} | {fw_m[1]:<8.1%} | {fw_m[2]:.1%}\n")
                self.output_file.flush()  # Force write to disk after each file
        
        # Calculate and display summary statistics
        print("\n" + "="*85)
        print("SUMMARY STATISTICS (User A only)")
        print("="*85)
        
        if baseline_tprs:
            bl_tpr_mean = np.mean(baseline_tprs)
            bl_tpr_std = np.std(baseline_tprs, ddof=1) if len(baseline_tprs) > 1 else 0
            bl_fnr_mean = np.mean(baseline_fnrs)
            
            print(f"Baseline   - TPR: {bl_tpr_mean:.1%} ± {bl_tpr_std:.1%}")
            print(f"Baseline   - FNR: {bl_fnr_mean:.1%}")
        
        if framework_tprs:
            fw_tpr_mean = np.mean(framework_tprs)
            fw_tpr_std = np.std(framework_tprs, ddof=1) if len(framework_tprs) > 1 else 0
            fw_fnr_mean = np.mean(framework_fnrs)
            
            print(f"Framework  - TPR: {fw_tpr_mean:.1%} ± {fw_tpr_std:.1%}")
            print(f"Framework  - FNR: {fw_fnr_mean:.1%}")
        
        # Write final block to file
        self.output_file.write("\n" + "="*90 + "\n")
        self.output_file.write("SUMMARY STATISTICS (User A only)\n")
        self.output_file.write("="*90 + "\n")
        if baseline_tprs:
            self.output_file.write(f"Baseline   - TPR: {bl_tpr_mean:.1%} ± {bl_tpr_std:.1%}\n")
        if framework_tprs:
            self.output_file.write(f"Framework  - TPR: {fw_tpr_mean:.1%} ± {fw_tpr_std:.1%}\n")
        
        self.output_file.close()
        print(f"\n✓ Detailed results saved to: {output_path}")

if __name__ == "__main__":
    evaluator = InfoGapEvaluator()
    evaluator.run_eval()