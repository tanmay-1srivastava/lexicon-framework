# Lexicon Framework

A framework for context-aware conversational AI with privacy-preserving relationship detection.

## 🚀 Quick Start Guide

### 1️⃣ Dataset Generation

Generate synthetic conversational datasets with context aggregation and info gap detection tasks.

#### Files to Run:

**Single Dataset Generation (Latest - Recommended)**
```bash
cd data_generation/new_data
python test_and_validate.py
```
- Generates a single dataset with quality validation
- Outputs timestamped file in `generated_datasets/`
- Validates context aggregation coverage and info gap detection

**Batch Dataset Generation**
```bash
cd data_generation/new_data
python batch_generate.py
```
- Generates 9 datasets across 3 scenario categories:
  - Friends Meeting (3 variations)
  - Work Collaboration (3 variations)
  - Doctor Visit (3 variations)

**Core Generation Module**
```bash
cd data_generation/new_data
python generate_dataset_azure.py
```
- Core Azure OpenAI-based dataset generation
- Uses prompts from `prompt.txt`
- Can be imported and used programmatically

---

### 2️⃣ Relationship Identification (One-Sided Analysis)

Estimate relationship types and privacy states from one-sided conversation analysis.

#### Files to Run:

**Persona Vector Analysis**
```bash
cd relationship_manager
python identify_realtion.py
```
- Analyzes one-sided conversations to build multi-dimensional persona vectors
- Scores: Formality (1-10), Trust/Proximity (1-10), Openness State
- Identifies relationship type and assigns access tiers
- Enriches datasets with persona information

**One-Sided Relationship Evaluation**
```bash
cd relationship_manager
python evaluate_relation.py
```
- Evaluates relationship estimation accuracy
- Predicts relationship from User A's linguistic signals only
- Focuses on:
  - Deictic Masking (hiding info with "it", "that thing")
  - Formality (titles vs. fragments)
  - Trust markers ("we", shared private spaces)
- Outputs tier and state accuracy metrics

---

## 📁 Project Structure

```
lexicon_framework/
├── data_generation/           # Dataset generation pipeline
│   └── new_data/
│       ├── generate_dataset_azure.py  # Core generation engine
│       ├── batch_generate.py          # Batch processing
│       ├── test_and_validate.py       # Single dataset + validation
│       └── prompt.txt                 # Generation prompts
│
├── relationship_manager/      # One-sided relationship estimation
│   ├── identify_realtion.py   # Persona vector analysis
│   └── evaluate_relation.py   # Evaluation & prediction
│
├── baselines/                 # Baseline methods (CoT, self-reflection)
├── context_aggregation/       # Context resolution methods
├── evaluation/                # Analysis and plotting scripts
└── info-gap-detection/        # Information gap detection
```

---

## 🔑 Configuration

Both dataset generation and relationship identification require Azure OpenAI credentials in `secret_keys.py`:

```python
Open_ai_key = "your-azure-openai-key"
```

---

## 📊 Output Files

**Dataset Generation:**
- `data_generation/new_data/generated_datasets/*.json`
- Contains: conversation transcript, context resolutions, protocol queries

**Relationship Analysis:**
- `data_generation/new_data/persona_enriched_datasets/*_persona.json`
- Adds: persona vector, access tier, answering protocol

---

## 🛠️ Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `openai` (Azure OpenAI SDK)
- `numpy`
- Standard Python libraries (json, datetime, os)
