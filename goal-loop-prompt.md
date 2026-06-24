# MindForge Build -- Orchestrator Goal Loop Prompt

## MISSION

Build MindForge: a Python CLI pipeline that probes LLMs for knowledge, scores correctness, corrects wrong answers, and outputs DPO training data -- running natively on Apple Silicon with MLX. Continue building until the full desktop app is finished.

## EXECUTION MODEL

You are the ORCHESTRATOR. Use cmux to open a workspace called "MindForge" and manage multiple Hermes agent panes inside it. You are the coordinator -- you do not write code yourself. You delegate to agent panes and verify their output.

## CMUX SETUP (Do This First)

```bash
cmux ping
cmux new-workspace --name "MindForge" --cwd /Users/cyber521k/MindForge --focus true
cmux tree --all
```

Create panes for each agent as needed. Use `cmux send` to dispatch commands to panes, `cmux read-screen` to verify results, and `cmux notify` when milestones complete.

## BUILD PHASES (Execute In Order, Do Not Skip Ahead)

### PHASE 1: Foundation (Python CLI -- No UI)

**Goal:** `mindforge probe --model X --subject Y` outputs valid DPO JSONL.

1. Create the project structure:
   ```
   mindforge/
   ├── README.md
   ├── requirements.txt          # mlx-lm, mlx-lm-lora, openai, requests, pymupdf, beautifulsoup4
   ├── taxonomy/
   │   └── subjects.yaml         # Full MMLU 57-subject taxonomy (STEM, Humanities, Social Science, Professional, Other)
   ├── mindforge/
   │   ├── __init__.py
   │   ├── cli.py                # Entry point: argparse CLI with subcommands
   │   ├── hardware/
   │   │   ├── detector.py       # detect_hardware() -- sysctl chip, memory, model
   │   │   └── api_keys.py       # detect_available_apis() -- scan env vars for 7 providers
   │   ├── probe/
   │   │   ├── engine.py         # ProbeEngine -- runs 3-tier probing (Tier 1 only for v1)
   │   │   ├── adapters.py       # ModelAdapter base + MLXAdapter, OpenAIAdapter, OpenRouterAdapter
   │   │   └── question_gen.py   # Question generation per subject per tier
   │   ├── score/
   │   │   ├── judge.py          # LLM-as-Judge scoring
   │   │   ├── answer_key.py     # MMLU answer key loader (ground truth for 57 subjects)
   │   │   └── confidence.py     # Confidence scoring (0.0-1.0)
   │   ├── correct/
   │   │   ├── analyzer.py       # Error analysis (what went wrong)
   │   │   └── corrector.py      # Corrected answer formulation
   │   ├── format/
   │   │   ├── dpo.py            # DPO formatter (DEFAULT)
   │   │   ├── alpaca.py         # Alpaca formatter
   │   │   ├── chatml.py         # ChatML formatter
   │   │   └── completion.py     # Completion formatter
   │   └── vault/
   │       ├── database.py       # SQLite storage layer (responses, training_entries, review_sessions tables)
   │       └── review.py         # In-CLI review prompt (Accept/Reject/Edit/Skip -- simple text, not TUI)
   ├── data/                     # gitignored
   │   ├── mindforge.db
   │   └── training-data/
   │       └── dpo/
   └── tests/
   ```

2. Key decisions for Phase 1:
   - Use MMLU answer keys as ground truth for the first domain (STEM Mathematics). Do NOT rely on LLM-as-Judge alone. Load answer keys from the MMLU dataset.
   - Start with Llama 3.2 3B 4-bit (`mlx-community/Llama-3.2-3B-Instruct-4bit`) as the default probe model. It's fast on any Apple Silicon Mac.
   - The CLI must work end-to-end: `mindforge probe --model mlx-community/Llama-3.2-3B-Instruct-4bit --subject mathematics --tier 1`
   - Output: valid DPO JSONL at `data/training-data/dpo/train.jsonl`
   - The review step is a simple CLI prompt (Accept/Reject/Edit/Skip), NOT a TUI dashboard.

3. Success criteria for Phase 1:
   - `pip install -e .` works
   - `mindforge probe --model mlx-community/Llama-3.2-3B-Instruct-4bit --subject mathematics` runs end-to-end
   - Produces `train.jsonl` with valid DPO format: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
   - At least 20 samples in the output
   - The output loads successfully in mlx-lm-lora (verify with a dry run)

### PHASE 2: Harden the Pipeline

**Goal:** Make Phase 1 robust and add scoring layers.

1. Add LLM-as-Judge as a fallback when no answer key exists (Layer 2 of scoring)
2. Add confidence scoring (0.0-1.0) with auto-approve threshold at 0.7
3. Add the correction pipeline (analyze error, look up truth, formulate corrected answer)
4. Add all 6 training format outputs (DPO default, Alpaca, ChatML, OpenAI Messages, Completion, Template-Free)
5. Add Tier 2 (depth) and Tier 3 (edge cases) probing -- toggleable via CLI flags
6. Add hardware auto-detection to the CLI (`mindforge detect` shows hardware + available models)
7. Add API key detection and cloud model adapters (OpenAI, OpenRouter)
8. Add SQLite database for persistent storage across runs

### PHASE 3: Model Conversion & Quantization

**Goal:** `mindforge convert` and `mindforge quantize` subcommands.

1. Implement model conversion: `mindforge convert --source mistralai/Mistral-7B-Instruct-v0.3 --quantize 4bit`
2. Implement standalone quantization: `mindforge quantize --model ~/mindforge-data/models/Mistral-7B-MLX --bits 4`
3. Support 2/3/4/6/8-bit and full precision
4. Track converted/quantized models in SQLite
5. Converted models appear in `mindforge detect` model list

### PHASE 4: Fine-Tuning & Evaluation

**Goal:** `mindforge train` and `mindforge evaluate` subcommands.

1. Implement DPO training via mlx-lm-lora: `mindforge train --model mlx-community/Llama-3.2-3B-Instruct-4bit --data data/training-data/dpo/ --mode dpo`
2. Support SFT, DPO, ORPO training modes
3. Implement evaluation via lm-eval-harness: `mindforge evaluate --model ./models/specialist-v1 --tasks mmlu_stem`
4. Before/after comparison (base vs. fine-tuned)
5. Training progress output (loss, iteration count, ETA)
6. Results stored in SQLite

### PHASE 5: PDF & Web Ingestion

**Goal:** `mindforge ingest-pdf` and `mindforge ingest-web` subcommands.

1. PDF ingestion: extract text, chunk, generate Q&A pairs, output DPO JSONL
2. Web URL ingestion: extract content with 3-layer anti-prompt-injection sanitization
3. Web extraction tiered: Firecrawl -> Crawl4AI -> Trafilatura -> BeautifulSoup
4. All ingested content goes through the same review + format pipeline
5. Track sources in SQLite (web_sources, pdf_sources tables)

### PHASE 6: Exo Cluster Integration

**Goal:** Auto-detect exo and use it for distributed inference.

1. Detect exo on startup (API check -> process check -> install check)
2. If exo active: use ExoAdapter, show cluster topology, enable larger models
3. If exo not active: silent fallback to single-device MLX
4. Exo for fine-tuning: distributed training with per-device monitoring
5. Cluster memory replaces single-device memory in model list

### PHASE 7: Desktop App (Tauri + React)

**Goal:** Replace the CLI with the Hermes-themed desktop app.

1. Set up Tauri 2 project structure (src-tauri/ + src/)
2. Build Python FastAPI sidecar (wraps the CLI as a WebSocket server)
3. Implement Hermes theme (gold #FFD700 on deep teal #041C1C, caduceus ⚕ logo)
4. Build Screen 1: Model Setup (hardware detection + model list)
5. Build Screen 2: Domain Setup (browseable subject catalog)
6. Build Screen 3: Probing Progress (live WebSocket updates)
7. Build Screen 4: Review Dashboard (Accept/Reject/Edit/Skip with keyboard shortcuts)
8. Build Screen 5: Format & Export (DPO default, live JSON preview)
9. Build Screen 6: Train & Evaluate (live loss curve, before/after table)
10. Build Screen 7: Stats Dashboard (accuracy by domain, model comparison)
11. Settings screen (theme, sound, API keys, data paths, probing config, web config, exo config)
12. Global keyboard shortcuts
13. Animated transitions, gold glow effects, optional sound
14. Package as .app bundle

### PHASE 8: Hermes Skill Wrapper

**Goal:** Wrap the desktop app as a Hermes Agent skill.

1. Create MindForge skill in ~/.hermes/skills/
2. Skill launches the desktop app and provides CLI access
3. Document all subcommands and features

## ORCHESTRATION RULES

1. **You are the orchestrator, not a coder.** Delegate to agent panes in cmux. Verify their output.
2. **One phase at a time.** Do not start Phase 2 until Phase 1 is verified working end-to-end.
3. **Verify every milestone.** Use `cmux read-screen` to check agent output. Run tests. Confirm files exist and code works before moving on.
4. **Use the design doc.** The full architecture spec is at /Users/cyber521k/MindForge/design-doc.md. Reference it for data schemas, UI mockups, API formats, and feature specs.
5. **Start small, prove, scale.** Llama 3.2 3B first. One subject first. DPO format first. Prove it works, then add features.
6. **MMLU answer keys first.** Do not rely on LLM-as-Judge alone for Phase 1. Use ground truth.
7. **Notify on milestones.** Use `cmux notify` when each phase completes.
8. **Never fabricate results.** If something doesn't work, report it honestly and fix it.
9. **Don't stop until the desktop app is finished.** The CLI (Phases 1-6) is the foundation. The desktop app (Phase 7) is the product. The Hermes skill (Phase 8) is the wrapper. All 8 phases must complete.

## REFERENCE DOCUMENTS

- Design doc: /Users/cyber521k/MindForge/design-doc.md
- cmux skill: loaded (use `cmux` commands for workspace/pane management)
- MLX models: https://huggingface.co/mlx-community
- mlx-lm-lora: https://github.com/Goekdeniz-Guelmez/mlx-lm-lora
- exo: https://github.com/exo-explore/exo
- MMLU dataset: https://huggingface.co/datasets/cais/mmlu
