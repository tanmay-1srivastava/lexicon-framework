import json

class HardLockEnforcer:
    def __init__(self):
        # TIER 0 REGISTRY: Static categories that bypass AI logic
        # These are non-negotiable hard rules.
        self.forbidden_registry = {
            "FINANCIAL": ["bank", "account", "routing", "balance", "credit", "debit", "wire", "transaction"],
            "LEGAL": ["power of attorney", "will", "proxy", "lawsuit", "legal", "notary"],
            "GOVERNMENT_ID": ["ssn", "social security", "passport", "driver's license", "government id"],
            "AUTHENTICATION": ["password", "pincode", "security question", "biometric"]
        }

    def check_gate(self, entity_name, relationship_tier):
        """
        The Deterministic Gate:
        Returns:
            - "ABORT": If Tier 0 violation occurs (Immediate Suppression).
            - "PROCEED": If the entity is safe for Fuzzy Logic processing.
        """
        entity_lower = entity_name.lower()
        
        # Step 1: Scan for Forbidden Keywords
        for category, keywords in self.forbidden_registry.items():
            if any(k in entity_lower for k in keywords):
                # Step 2: Enforcement Logic
                # Tier 0 data is ONLY accessible to Tier 1 (Spouse/Intimate)
                if relationship_tier != 1:
                    return {
                        "status": "ABORT",
                        "category": category,
                        "reason": f"Deterministic Block: {category} data is restricted to Tier 1 relationships."
                    }
                else:
                    # Even for Tier 1, we log the high-sensitivity access
                    return {
                        "status": "PROCEED", 
                        "category": category, 
                        "note": "Authorized Tier 1 access to sensitive category."
                    }

        # Step 3: If no forbidden keywords found, hand off to Fuzzy Logic Matrix
        return {"status": "PROCEED", "category": "GENERAL_CLINICAL"}

# --- MODULAR TEST CASE ---
if __name__ == "__main__":
    enforcer = HardLockEnforcer()
    
    # Scenario A: Coworker (Tier 3) asking for bank info
    result_a = enforcer.check_gate("Bank account balance", relationship_tier=3)
    print(f"Scenario A: {result_a['status']} - {result_a.get('reason', 'Safe')}")
    
    # Scenario B: Spouse (Tier 1) asking for bank info
    result_b = enforcer.check_gate("Bank account balance", relationship_tier=1)
    print(f"Scenario B: {result_b['status']} - {result_b.get('note', 'Safe')}")
    
    # Scenario C: Coworker (Tier 3) asking for a Medication (Not a Hard Lock keyword)
    result_c = enforcer.check_gate("Methotrexate dosage", relationship_tier=3)
    print(f"Scenario C: {result_c['status']} -> Handing off to Fuzzy Matrix")