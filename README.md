# MindForge

AI model probing and correction system for generating DPO training data from MMLU evaluations.

## Overview

MindForge probes language models against MMLU (Massive Multitask Language Understanding) questions, identifies incorrect answers, and generates corrected DPO (Direct Preference Optimization) training pairs.

## Installation

```bash
cd MindForge
pip install -e .
```

## Usage

### Detect Hardware

```bash
mindforge detect
```

Shows Apple Silicon chip info, memory, and available API keys.

### Probe a Model

```bash
mindforge probe --model mlx-community/Llama-3.2-3B-Instruct-4bit --subject mathematics
```

Probes the model against MMLU questions and generates DPO training data at `data/training-data/dpo/train.jsonl`.

Options:
- `--model` : MLX model name (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
- `--subject` : Subject to probe (default: mathematics)
- `--tier` : Probing tier, 1-3 (default: 1)
- `--limit` : Number of questions (default: 25)

### Review Training Entries

```bash
mindforge review
```

Interactive review of generated training pairs (Accept/Reject/Edit/Skip).

### Format Output

```bash
mindforge format --input data/responses.json --format dpo --output data/training-data/dpo/train.jsonl
```

## Supported Subjects

All 57 MMLU subjects across STEM, Humanities, Social Science, Professional, and Other categories.

## Output Format

DPO JSONL format, one JSON object per line:
```json
{"prompt": "Question text\nA) ... B) ... C) ... D) ...", "chosen": "The answer is B) ...", "rejected": "The answer is A) ..."}
```
