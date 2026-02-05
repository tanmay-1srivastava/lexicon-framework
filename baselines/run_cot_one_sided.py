#!/usr/bin/env python3
"""
Run CoT Basic with ONE-SIDED conversation (User A only) - fair comparison with improved approach
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI
import sys
sys.path.insert(0, '/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework')
from secret_keys import Open_ai_key as azure_openai_key

# Config
IN_DIR = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/data_generation/new_data/generated_datasets"
OUT_DIR = "/Users/tanmay-s/Desktop/Internships/msr_25/lexicon_framework/baselines/cot_one_sided_results"

AZURE_OPENAI_ENDPOINT = "https://initial-resources.cognitiveservices.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

SYSTEM_INSTRUCTIONS = """
You are an expert at resolving ambiguous spatial and temporal references using provided context.

IMPORTANT:
- You only see ONE SIDE of the conversation (User A's messages)
- You do NOT know what the other person said
- Output ONLY valid JSON (no markdown, no commentary)
- Focus ONLY on spatial and temporal references

Your output MUST be a JSON object with "ground_truth_resolutions" key.
Each resolution must have:
{
  "trigger_turn_id": int,
  "ambiguous_phrase": "",
  "resolved_entity": "",
  "resolution_source": ""
}

Resolution sources: "User A GPS + Wifi", "User A Calendar", "Conversation Context"
""".strip()

OUTPUT_SCHEMA = {
    "name": "GroundTruthResolutionsLikeOutput",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ground_truth_resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "trigger_turn_id": {"type": "integer"},
                        "ambiguous_phrase": {"type": "string"},
                        "resolved_entity": {"type": "string"},
                        "resolution_source": {"type": "string"},
                    },
                    "required": [
                        "trigger_turn_id",
                        "ambiguous_phrase",
                        "resolved_entity",
                        "resolution_source",
                    ],
                },
            }
        },
        "required": ["ground_truth_resolutions"],
    },
}


def filter_user_a_only(transcript):
    """Keep only User A's messages"""
    return [t for t in transcript if t.get('speaker') == 'User A']


def call_azure_chat(client, deployment, user_payload, max_tokens=4096, temperature=0.2, retries=5):
    """Call Azure OpenAI with retries"""
    
    user_task = {
        "task": "Detect spatial and temporal ambiguous phrases in User A's messages and resolve them using metadata.",
        "instructions": [
            "ONLY resolve spatial references (here, there, this room, server room, etc.)",
            "ONLY resolve temporal references (now, then, tomorrow, later, etc.)",
            "Use metadata (location_semantic, GPS, calendar_next) to resolve",
            "Skip person/object references (it, he, she, them)",
            "You only see User A's messages - infer context from metadata",
            "Return JSON with ground_truth_resolutions list"
        ],
        "input": user_payload,
    }

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(user_task, ensure_ascii=False)},
    ]

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_schema", "json_schema": OUTPUT_SCHEMA},
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)

            if "ground_truth_resolutions" not in parsed:
                raise ValueError("Missing ground_truth_resolutions")

            return parsed

        except Exception as e:
            last_err = e
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))

    raise RuntimeError(f"Failed after {retries} retries. Last error: {last_err}")


def main():
    client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=azure_openai_key,
    )

    in_path = Path(IN_DIR)
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob("*.json"))
    
    print(f"Processing {len(files)} files with ONE-SIDED conversation (User A only)...\n")

    for fp in files:
        print(f"Processing {fp.name}...", end=" ")
        
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            
            # Filter to User A only
            full_transcript = data.get('conversation_transcript', [])
            user_a_transcript = filter_user_a_only(full_transcript)
            
            # Build one-sided payload
            payload = {
                "backstory": data.get("backstory", {}),
                "mobile_context_snapshot": {
                    "user_a": data.get("mobile_context_snapshot", {}).get("user_a", {})
                },
                "conversation_transcript": user_a_transcript  # ONE-SIDED
            }
            
            # Call LLM
            result = call_azure_chat(
                client=client,
                deployment=AZURE_OPENAI_DEPLOYMENT,
                user_payload=payload,
            )
            
            # Save output
            out_obj = {
                "dataset_id": data.get("dataset_id", ""),
                "file": fp.name,
                "mode": "one_sided_user_a",
                "ground_truth_resolutions": result["ground_truth_resolutions"],
            }
            
            out_file = out_path / (fp.stem + "_cot_one_sided_result.json")
            with open(out_file, 'w') as f:
                json.dump(out_obj, f, indent=2)
            
            print(f"✓ ({len(result['ground_truth_resolutions'])} resolutions)")
            
        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
