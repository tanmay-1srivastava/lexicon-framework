from hard_lock_enforcer import HardLockEnforcer
from fuzzy_logic_matrix import FuzzyLogicMatrix
from resolution_executor import ResolutionExecutor

# --- TEST DATASET ---
scenarios = [
    {
        "name": "Spouse accessing Bank Info",
        "entity": "Bank Account Balance",
        "tier": 1,
        "transcript": [{"speaker": "User A", "text": "Honey, check the savings balance."}]
    },
    {
        "name": "Coworker accessing Bank Info",
        "entity": "Bank Routing Number",
        "tier": 3,
        "transcript": [{"speaker": "User A", "text": "Let's finish the project by 5."}]
    },
    {
        "name": "Friend accessing Medication (Open State)",
        "entity": "Insulin Dosage",
        "tier": 2,
        "transcript": [{"speaker": "User A", "text": "I need to take my insulin, can you find the box?"}]
    },
    {
        "name": "Friend accessing Medication (Closed State)",
        "entity": "Insulin Dosage",
        "tier": 2,
        "transcript": [{"speaker": "User A", "text": "I have to take 'it' now. Please wait outside for a second."}]
    },
    {
        "name": "Casual Staff accessing Appointment Time",
        "entity": "Follow-up Appointment Time",
        "tier": 3,
        "transcript": [{"speaker": "User A", "text": "I'll see you at the front desk."}]
    }
]

def run_integration_test():
    enforcer = HardLockEnforcer()
    matrix = FuzzyLogicMatrix()
    executor = ResolutionExecutor()
    
    # Mock Mobile Context
    mock_context = {"bank_balance": "$4,000", "insulin_dosage": "10 units", "appt_time": "2:00 PM"}

    print(f"{'SCENARIO':<35} | {'GATE RESULT':<15} | {'FINAL ACTION'}")
    print("-" * 75)

    for case in scenarios:
        # 1. HARD LOCK (Tier 0 Check)
        gate_check = enforcer.check_gate(case['entity'], case['tier'])
        
        if gate_check['status'] == "ABORT":
            decision = "SUPPRESS"
            reason = gate_check['reason']
        else:
            # 2. FUZZY MATRIX (Social & State Check)
            social_eval = matrix.enforce_social_norms(case['entity'], case['transcript'], case['tier'])
            decision = social_eval['action']
            reason = social_eval['reason']

        # 3. EXECUTION (Output Generation)
        final_output = executor.execute_protocol(decision, {"entity": case['entity']}, mock_context)
        
        print(f"{case['name']:<35} | {decision:<15} | {final_output.get('type', 'RESTRICTED')}")

if __name__ == "__main__":
    run_integration_test()