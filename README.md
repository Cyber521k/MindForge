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
├── tests/             # 14 test files (phases 1-8, functional, E2E, edge cases, round 2, domain expansion, Xbox blades)
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
- `--subject` : Subject to probe (default: mathematics). Supports aliases (e.g. `py`, `js`, `k8s`, `aws`).
- `--tier`    : Probing tier, 1-3 (default: 1)
- `--limit`   : Number of questions (default: 25)

Available subjects: 110 across 10 categories (57 MMLU + 53 extended). Run `mindforge models` or see `taxonomy/subjects.yaml` for the full list.

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

The Tauri desktop app provides a GUI for all MindForge operations with an authentic Xbox Blades UI inspired by the original Xbox dashboard.

### Xbox Blades UI

The interface recreates the original Xbox dashboard experience with horizontal blade tabs at the bottom of the screen, 3D perspective transitions, frosted glass panels, and programmatic sound effects.

#### Layout

- **Top bar**: Caduceus logo, model name, connection status, mute toggle
- **Upper 2/3**: Blade content area with 3D perspective (`perspective: 1200px`), hexagonal grid background, and subtle scanline overlay
- **Bottom**: Horizontal blade bar (`BladeBar.tsx`) with 8 blade tabs

#### Blade Bar (BladeBar.tsx)

Horizontal tabs at the bottom of the screen. The active blade expands upward to fill the content area:

- Active blade: full height, gold gradient, frosted glass (`backdropFilter: blur(8px)`), gold glow, angled clip-path edges, gold accent indicator bar
- Inactive blades: 70% height, dimmed, subtle border
- Spring animation on height/position transitions (stiffness: 300, damping: 25)
- Plays "sweep" sound on blade change

#### Blade Content (BladeContent.tsx)

Each screen is wrapped in a two-panel layout:

- **Left panel (32%)**: Large decorative icon (96px, pulsing glow animation), screen title (uppercase, gold, letter-spaced), decorative gradient line
- **Right panel (68%)**: Scrollable content area for the screen component
- Frosted glass backgrounds (`backdropFilter: blur(12px)`)
- Radial spotlight gradient on the background

#### Blade Transitions

- **Sweep animation**: Screens slide horizontally with `rotateY: 15deg` + `translateX` via Framer Motion custom variants
- **Direction-aware**: Forward navigation sweeps right-to-left, back sweeps left-to-right
- **Spring physics**: stiffness 260, damping 28 for natural movement
- `AnimatePresence mode="wait"` prevents overlapping blades
- `aria-live="polite"` announces content changes to screen readers

#### Sound Effects (SoundManager.tsx)

All sounds are generated programmatically via Web Audio API -- no audio files needed:

| Sound | Trigger | Description |
|-------|---------|-------------|
| Sweep | Blade change | Filtered noise burst with pitch sweep (bandpass 400Hz->2000Hz, 350ms) |
| Select | Menu item selection | Sine wave ping at 800Hz, 50ms decay |
| Scroll | Navigation scroll | Square wave tick at 1200Hz, 20ms |
| Back | Back navigation | Reverse whoosh (pitch sweeps 2000Hz->200Hz, 300ms) |
| Ambient | Optional background | Low drone at 55Hz with LFO modulation (toggleable) |

- Mute toggle in top bar (persists to localStorage)
- `SoundEngine` singleton via `getSoundEngine()`

#### Visual Effects (index.css)

- **Hexagonal grid**: Three repeating linear gradients at 60deg/-60deg/0deg with radial spotlight (rgba gold, 3% opacity)
- **Scanlines**: Subtle CRT effect (2px transparent, 1px rgba black 8%, 50% opacity)
- **Frosted glass**: `.frosted-glass` class -- rgba surface with 12px backdrop blur
- **Xbox menu items**: `.xbox-menu-item` / `.xbox-menu-item-active` -- gold left border, gradient highlight on active

#### Navigation

- **ArrowLeft/ArrowRight**: Switch between blades (disabled on Review Dashboard -- arrows navigate queue items)
- **Click blade tab**: Navigate to that blade
- **Controller hints**: "Left/Right = Navigate" (bottom left), "Enter = Select" (bottom right)

#### Accessibility

- `aria-live="polite"` on main content area
- `role="tab"` and `aria-selected` on blade tabs
- `aria-label` on all interactive elements
- `prefers-reduced-motion` disables all animations
- Keyboard navigation (Enter/Space on blade tabs)
- Skip-to-content link for screen reader users

### Screens

| # | Blade | Icon | Description |
|---|-------|------|-------------|
| 1 | Model Setup | Monitor | Select and configure models |
| 2 | Domain Setup | Book | Choose MMLU subjects and probing tier |
| 3 | Probe Engine | Magnifier | Real-time progress with WebSocket streaming |
| 4 | Review Dashboard | Clipboard | Accept/Reject/Edit training pairs with keyboard shortcuts |
| 5 | Format & Export | Package | Export in 6 formats |
| 6 | Train & Evaluate | Target | Launch training and evaluation jobs |
| 7 | Statistics | Chart | 5 SVG/CSS charts (bar, pie, line, progress, cards) |
| 8 | Settings | Gear | Theme, animations, auto-approve threshold, sound toggle |

### Sidecar Lifecycle

The Rust backend (`src-tauri/src/main.rs`) provides IPC commands for managing the Python sidecar process:

- `start_sidecar` - Launch the FastAPI server process
- `stop_sidecar` - Kill the sidecar process
- `sidecar_status` - Check if the sidecar is running

## Supported Subjects

110 subjects across 10 categories. See `taxonomy/subjects.yaml` for the full list.

### Original MMLU Categories (57 subjects)

| Category | Count | Examples |
|----------|-------|----------|
| STEM | 14 | mathematics, physics, chemistry, biology, computer science |
| Humanities | 12 | history, philosophy, moral disputes, world religions |
| Social Science | 11 | economics, psychology, sociology, geography |
| Professional | 13 | law, medicine, accounting, electrical engineering |
| Other | 7 | anatomy, marketing, nutrition, professional psychology |

### Extended Categories (53 subjects)

| Category | Count | Examples |
|----------|-------|----------|
| Agent Frameworks | 12 | hermes_agent, langchain, autogen, crewai, babyagi |
| Programming Languages | 16 | python, rust, go, typescript, swift, zig, nim |
| Blockchain/Web3 | 14 | solidity, solana, cosmos, defi, nft, foundry |
| DevOps/Infrastructure | 7 | docker, kubernetes, terraform, AWS, GCP, Azure |
| Security/Cryptography | 4 | cryptography, secure coding, pentesting, network security |

### Subject Aliases

Common abbreviations are mapped automatically. Examples: `py`->python, `js`->javascript, `ts`->typescript, `c++`->cpp, `c#`->csharp, `k8s`->kubernetes, `tf`->terraform, `aws`->cloud_aws, `gcp`->cloud_gcp, `azure`->cloud_azure, `crypto`->cryptography, `pentest`->pentesting, `netsec`->network_security, `hermes`->hermes_agent.

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

# Run domain expansion tests (new subjects, aliases, taxonomy)
python -m pytest tests/test_domain_expansion.py -v

# Run Xbox Blades UI tests (transitions, sound, navigation)
python -m pytest tests/test_xbox_blades.py -v
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
├── tests/               # 14 test files (727 tests)
├── taxonomy/            # MMLU subject taxonomy
├── setup.py             # Python package config
├── package.json         # Node package config
└── .github/workflows/   # CI pipeline
```
