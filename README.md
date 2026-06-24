# MindForge

AI model probing and correction system for generating DPO training data from MMLU evaluations.

## Overview

MindForge probes language models against MMLU (Massive Multitask Language Understanding) questions, identifies incorrect answers, and generates corrected DPO (Direct Preference Optimization) training pairs. It includes a CLI, a FastAPI sidecar server, a Tauri desktop app (React + Rust), and a full pipeline for ingesting PDFs and web content into training data.

## Architecture

```
MindForge/
├── mindforge/          # Python package (CLI + pipeline modules)
│   ├── __main__.py    # Entry point: `python -m mindforge`
│   ├── cli.py         # CLI with 11 commands
│   ├── hardware/      # Hardware detection (Apple Silicon, API keys, exo, Ollama)
│   ├── probe/         # MMLU probing engine and adapters
│   ├── score/         # Answer scoring, confidence, judge
│   ├── correct/       # Analyzer + corrector for DPO pair generation
│   ├── format/        # Output formatters (DPO, Alpaca, ChatML, completion, etc.)
│   ├── convert/       # MLX model conversion + quantization
│   ├── train/         # LoRA/DPO/ORPO training via mlx-lm-lora
│   ├── evaluate/      # Model evaluation (lm-eval-harness, mlx)
│   ├── ingest/        # PDF/web extraction, sanitization, Q&A generation
│   └── vault/         # SQLite database + review system
├── python/server.py   # FastAPI sidecar server (20 routes, WebSocket streaming)
├── src/               # React frontend (Tauri webview)
│   ├── screens/       # 8 screen components
│   └── components/    # Reusable UI (ErrorState, LoadingState, etc.)
├── src-tauri/         # Rust Tauri backend (sidecar lifecycle management)
├── tests/             # 12 test files (phases 1-8, functional, E2E, edge cases, round 2 features)
├── taxonomy/           # MMLU subject taxonomy (57 subjects, 5 domains)
└── .github/workflows/ # CI pipeline
```

## Installation

### Prerequisites

- Python 3.11+
- Node.js 20+ and pnpm
- Rust toolchain (for Tauri desktop app)
- Apple Silicon Mac (for MLX inference/training)

### Python Package

```bash
cd MindForge
pip install -e .
```

### Desktop App (Tauri)

```bash
pnpm install
pnpm tauri dev    # development
pnpm tauri build  # production build
```

## CLI Usage

### Detect Hardware

```bash
mindforge detect
```

Shows Apple Silicon chip info, memory, available API keys, exo cluster status, and Ollama models.

Options:
- `--verbose` : Show detailed hardware info including exo cluster nodes
- `--quiet`   : Show minimal output (machine-readable)

### List Models

```bash
mindforge models
```

Lists available local (MLX, Ollama) and cloud (OpenAI, Anthropic, OpenRouter) models.

### Probe a Model

```bash
mindforge probe --model mlx-community/Llama-3.2-3B-Instruct-4bit --subject mathematics
```

Probes the model against MMLU questions and generates DPO training data at `data/training-data/dpo/train.jsonl`.

Options:
- `--model`   : MLX model name (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
- `--subject` : Subject to probe (default: mathematics)
- `--tier`    : Probing tier, 1-3 (default: 1)
- `--limit`   : Number of questions (default: 25)

### Review Training Entries

```bash
mindforge review
```

Interactive review of generated training pairs (Accept / Reject / Edit / Skip).

### Format Output

```bash
mindforge format --input data/responses.json --format dpo --output data/training-data/dpo/train.jsonl
```

Supported formats: `dpo`, `alpaca`, `chatml`, `completion`, `openai_messages`, `template_free`

### Convert / Quantize Models

```bash
mindforge convert --source mlx-community/Llama-3.2-3B-Instruct-4bit --quantize 4bit
mindforge quantize --model ./my-model --bits 4
```

### Train

```bash
mindforge train --model mlx-community/Llama-3.2-3B-Instruct-4bit --data data/training-data/dpo/train.jsonl --method dpo
```

Supports LoRA, DPO, and ORPO training via mlx-lm-lora.

### Evaluate

```bash
mindforge evaluate --model ./my-model --benchmark mmlu --limit 100
```

### Ingest PDF

```bash
mindforge ingest-pdf --file /path/to/document.pdf --subject biology
```

Extracts text from PDFs, generates Q&A pairs, and formats as DPO training data. Requires `pymupdf`.

### Ingest Web

```bash
mindforge ingest-web --url https://example.com/article
```

Extracts content from web pages, sanitizes against prompt injection, generates Q&A pairs, and formats as DPO training data. Requires `beautifulsoup4` and `requests`.

## FastAPI Server

The sidecar server (`python/server.py`) exposes 19 REST API routes and a WebSocket endpoint for real-time progress streaming.

### Running the Server

```bash
python python/server.py
```

Default port: 7878

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/hardware` | Hardware detection |
| GET  | `/api/models` | Available models |
| GET  | `/api/taxonomy` | MMLU subject taxonomy |
| GET  | `/api/responses` | Probe results (SQLite) |
| GET  | `/api/training-entries` | Training entries |
| POST | `/api/probe` | Start probing job |
| GET  | `/api/probe/{job_id}` | Get probe status |
| POST | `/api/review/{entry_id}` | Submit review action |
| POST | `/api/format` | Format training data |
| POST | `/api/convert` | Convert model |
| POST | `/api/quantize` | Quantize model |
| POST | `/api/train` | Start training job |
| POST | `/api/evaluate` | Start evaluation job |
| POST | `/api/ingest-pdf` | Ingest PDF document |
| POST | `/api/ingest-web` | Ingest web URL |
| GET  | `/api/stats` | Aggregate statistics |
| GET  | `/api/jobs` | List all jobs |
| GET  | `/api/jobs/{job_id}` | Get job status |
| POST | `/api/jobs/{job_id}/cancel` | Cancel a running job |
| GET  | `/api/jobs/{job_id}/result` | Get job result |
| WS   | `/ws` | WebSocket (progress streaming, heartbeat, subscribe) |

## Desktop App (Tauri)

The Tauri desktop app provides a GUI for all MindForge operations with an Xbox Blades-style UI inspired by the original Xbox dashboard.

### Xbox Blades UI

The interface uses a blade navigation system where each screen is a "blade" that sweeps in and out with 3D perspective transitions:

- **Blade sweep transitions** -- Screens slide horizontally with `rotateY` (45deg) and `translateX` via Framer Motion. Direction-aware: sweeping right-to-left when navigating forward, left-to-right when going back.
- **Perspective container** -- The app uses CSS `perspective: 1400px` with `preserve-3d` for depth.
- **Blade tabs** -- Vertical sidebar tabs with angled right edges (CSS `clip-path` polygon), gold gradient on active blade, inset glow, and a gold accent indicator bar.
- **Arrow key navigation** -- ArrowLeft/ArrowRight switch between blades. ArrowUp/ArrowDown reserved for in-screen content navigation.
- **Accessibility** -- `aria-live="polite"` on `<main>` announces content changes. `AnimatePresence mode="wait"` prevents focus trapping during transitions.
- **Reduced motion** -- `prefers-reduced-motion` media query disables all animations.

### Sound Effects (Planned)

The sound system is designed but not yet implemented. The planned audio feedback:

- **Whoosh** on blade change (300-400ms transition sound)
- **Click** on menu item selection
- **Tick** on scroll/navigation
- **Ambient background music** (toggleable in Settings)

Implementation will use Web Audio API or Howler.js, with all sounds toggleable via Settings.

### Controller Hints (Planned)

Bottom-corner controller hints matching the original Xbox dashboard:
- **A = Select** (bottom right)
- **B = Back** (bottom left)

### Screens

- **Model Setup** - Select and configure models
- **Domain Setup** - Choose MMLU subjects and probing tier
- **Probing Progress** - Real-time progress with WebSocket streaming
- **Review Dashboard** - Accept/Reject/Edit training pairs with keyboard shortcuts
- **Format Export** - Export in 6 formats
- **Train & Evaluate** - Launch training and evaluation jobs
- **Stats** - Aggregate accuracy and training statistics with 5 SVG/CSS charts (bar, pie, line, progress bars, summary cards)
- **Settings** - Theme, animations, auto-approve threshold, etc.

### Sidecar Lifecycle

The Rust backend (`src-tauri/src/main.rs`) provides IPC commands for managing the Python sidecar process:

- `start_sidecar` - Launch the FastAPI server process
- `stop_sidecar` - Kill the sidecar process
- `sidecar_status` - Check if the sidecar is running

## Supported Subjects

All 57 MMLU subjects across STEM, Humanities, Social Science, Professional, and Other categories. See `taxonomy/subjects.yaml` for the full list.

## Output Format

DPO JSONL format, one JSON object per line:

```json
{"prompt": "Question text\nA) ... B) ... C) ... D) ...", "chosen": "The answer is B) ...", "rejected": "The answer is A) ..."}
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific phase tests
python -m pytest tests/test_phase1.py -v

# Run functional tests (executes CLI via subprocess)
python -m pytest tests/test_functional.py -v

# Run E2E tests (full pipeline, FastAPI TestClient, WebSocket)
python -m pytest tests/test_e2e.py -v

# Run edge-case tests (error handling, boundary conditions)
python -m pytest tests/test_edge_cases.py -v

# Run round 2 feature tests (Ollama, async endpoints, DB indexes)
python -m pytest tests/test_round2_features.py -v
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push and PR to `main`:

- **Python tests** (macOS) - Installs the package and runs pytest
- **Frontend build** (Ubuntu) - Type-checks and builds the React/Tauri frontend

## Project Structure

```
MindForge/
├── mindforge/           # Python package
│   ├── cli.py           # CLI entry point (11 commands)
│   ├── hardware/        # Hardware detection + model listing
│   ├── probe/           # MMLU probing engine
│   ├── score/           # Answer scoring + confidence
│   ├── correct/         # DPO pair generation
│   ├── format/          # 6 output formatters
│   ├── convert/         # MLX conversion
│   ├── train/           # LoRA/DPO/ORPO training
│   ├── evaluate/        # Model evaluation
│   ├── ingest/          # PDF + web ingestion
│   └── vault/           # SQLite database + review
├── python/server.py     # FastAPI sidecar (20 routes)
├── src/                 # React frontend
├── src-tauri/           # Rust Tauri backend
├── tests/               # 12 test files (567 tests)
├── taxonomy/            # MMLU subject taxonomy
├── setup.py             # Python package config
├── package.json         # Node package config
└── .github/workflows/   # CI pipeline
```
