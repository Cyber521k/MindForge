# Changelog

All notable changes to MindForge are documented in this file.

## [7.2.0] - 2026-06-24 (Automated Review)

### Added

- **AutoReviewer** (`mindforge/review/auto_reviewer.py`): Automated review of training entries using LLM-as-judge + web search
  - Judge LLM evaluates chosen/rejected answers for correctness
  - Web search fallback (DuckDuckGo Lite, no API key) when judge confidence < 0.7
  - Auto-detects judge model: OpenAI > OpenRouter > Ollama > MLX
  - Returns action: accept / reject / edit with confidence score
  - Batch review with progress callback
- **CLI auto-review flags**:
  - `--auto`: Enable automated review mode (default: manual interactive)
  - `--judge-model`: Specify judge model (e.g., gpt-4o, openrouter/anthropic/claude-3.5-sonnet)
  - `--no-web`: Disable web search fallback for uncertain answers
- **`auto_review_session()`** in `mindforge/vault/review.py`: Runs AutoReviewer on pending entries, applies results to DB
- **3 new FastAPI routes**:
  - `POST /api/review/auto` -- Start automated review job (streams progress via WebSocket)
  - `GET /api/review/auto/{job_id}` -- Get auto-review job status
  - `POST /api/review/auto/entry/{entry_id}` -- Auto-review a single entry (synchronous)
- **2 new taxonomy routes**:
  - `GET /api/taxonomy/search` -- Search taxonomy by query
  - `GET /api/taxonomy/{domain}` -- Get subjects for a specific domain

### Changed

- REST API routes: 19 -> 24 (5 new routes)
- Total endpoints: 20 -> 25 (24 REST + 1 WebSocket)
- Review command: now supports both manual and automated modes
- Route count in docs updated accordingly

## [7.1.0] - 2026-06-24 (Domain Expansion)

### Added

- **5 new subject categories** (53 new subjects, 110 total):
  - Agent Frameworks (12): hermes_agent, langchain, autogen, crewai, babyagi, etc.
  - Programming Languages (16): python, rust, go, typescript, swift, zig, nim, etc.
  - Blockchain/Web3 (14): solidity, solana, cosmos, defi, nft, foundry, etc.
  - DevOps/Infrastructure (7): docker, kubernetes, terraform, AWS, GCP, Azure
  - Security/Cryptography (4): cryptography, secure coding, pentesting, network security
- **48 subject aliases** (105 total mappings): py->python, js->javascript, ts->typescript, c++->cpp, c#->csharp, k8s->kubernetes, tf->terraform, cicd->ci_cd, aws->cloud_aws, gcp->cloud_gcp, azure->cloud_azure, crypto->cryptography, pentest->pentesting, netsec->network_security, hermes->hermes_agent

### Changed

- Subject count: 57 -> 110 (53 new extended subjects)
- Category count: 5 -> 10 (5 new extended categories)
- Subject mappings: 57 -> 105 (48 new aliases)
- taxonomy/subjects.yaml: added Agent_Frameworks, Programming_Languages, Blockchain_Web3, DevOps_Infrastructure, Security_Cryptography categories
- **Domain expansion tests**: 63 new tests (test_domain_expansion.py) -- new subjects, aliases, taxonomy validation
- **Xbox Blades tests**: 97 new tests (test_xbox_blades.py) -- blade transitions, sound effects, navigation, accessibility
- Test count: 567 -> 727 (160 new tests)
- Test files: 12 -> 14 (added test_domain_expansion.py, test_xbox_blades.py)

## [7.0.2] - 2026-06-24 (Round 3)

### Added

- **Xbox Blades UI redesign**: Complete recreation of the original Xbox dashboard experience
  - **BladeBar.tsx**: Horizontal blade tabs at bottom of screen. Active blade expands upward, frosted glass (backdrop-filter blur), gold gradient, angled clip-path edges, spring animations
  - **BladeContent.tsx**: Two-panel layout per blade -- left 32% decorative icon (96px, pulsing glow) + title, right 68% scrollable content. Frosted glass, radial spotlight gradient
  - **SoundManager.tsx**: Web Audio API sound engine (no audio files). Sweep (filtered noise, pitch sweep 400-2000Hz), select (sine 800Hz), scroll (square 1200Hz), back (reverse whoosh), ambient drone (55Hz + LFO). Mute toggle persists to localStorage
  - **Hexagonal grid background**: Three repeating linear gradients (60deg/-60deg/0deg) with radial spotlight (index.css `.hex-grid`)
  - **Scanline overlay**: Subtle CRT effect (index.css `.scanlines`)
  - **Frosted glass CSS**: `.frosted-glass` class with 12px backdrop blur
  - **Xbox menu item styles**: `.xbox-menu-item` / `.xbox-menu-item-active` with gold left border and gradient highlight
  - **Controller hints**: "Left/Right = Navigate" (bottom left), "Enter = Select" (bottom right)
  - **3D blade sweep transitions**: rotateY 15deg + translateX, direction-aware, spring physics (stiffness 260, damping 28)
  - **Arrow key navigation**: ArrowLeft/ArrowRight switch blades (disabled on Review Dashboard)
  - **Mute toggle**: Top bar button, persists to localStorage, unmute plays select sound
- **Round 2 feature tests**: 63 new tests (test_round2_features.py) -- Ollama integration, async endpoints, DB indexes, ErrorBoundary

### Changed

- Test count: 504 -> 567 (63 new round 2 feature tests)
- Test files: 11 -> 12 (added test_round2_features.py)
- App layout: sidebar + content -> top bar + blade content + bottom blade bar
- Screen transitions: simple slide -> 3D blade sweep with perspective (1200px) and rotateY
- Navigation: setScreen -> navigate() with direction tracking + sound playback
- Screens: wrapped in BladeContent (icon + title + content layout)
- CSS perspective: 1400px -> 1200px, perspective-origin: center -> center 40%

## [7.0.1] - 2026-06-24 (Round 2)

### Added

- **Ollama adapter**: `OllamaAdapter` in adapters.py for local Ollama models (ollama/ prefix, strips prefix before API call)
- **Ollama detector**: `mindforge/hardware/ollama_detector.py` — 3-tier detection (HTTP probe, pgrep, which)
- **Edge-case test suite**: `tests/test_edge_cases.py` — 89 tests covering error handling, boundary conditions, invalid inputs
- **WebSocket hook improvements**: reconnection with max retries, jitter, unmount guard

### Fixed

- Frontend: useMemo for screen elements (prevents re-instantiation on every render)
- FastAPI: blocking I/O wrapped in asyncio.to_thread() for 6 endpoints
- Database: added 6 indexes + WAL journal mode for query performance
- WebSocket: max reconnect attempts (20), jitter, unmount guard

### Changed

- Test count: 415 -> 504 (89 new edge-case tests)
- Test files: 10 -> 11 (added test_edge_cases.py)
- detect command: now shows Ollama models when running
- models command: now lists Ollama models with usage hint

## [0.0.1] - 2026-06-24

### Added

- **11 CLI commands**: detect, models, probe, review, format, convert, quantize, train, evaluate, ingest-pdf, ingest-web
- **FastAPI sidecar server** with 19 REST routes + WebSocket endpoint (port 7878)
  - Job management: create, cancel, status, result
  - WebSocket: heartbeat (30s), subscribe, reconnection
  - Input validation and structured error handling on all routes
  - Scoped CORS for localhost origins
  - Lifespan handler (replaces deprecated @app.on_event)
- **Tauri 2 desktop app** (.app bundle, 9.0 MB)
  - Rust IPC commands: start_sidecar, stop_sidecar, sidecar_status
  - Full icon set (macOS .icns, Windows .ico, iOS, Android)
- **React frontend** with 8 screens and 10 reusable components
  - Stats dashboard with 5 SVG/CSS charts (bar, pie, line, progress, cards)
  - Review dashboard with keyboard shortcuts and help overlay
  - Settings with API integration and retry
  - ErrorState and LoadingState reusable components
- **6 output formats**: DPO, Alpaca, ChatML, completion, openai_messages, template_free
- **PDF and web ingestion** with prompt injection sanitization
- **Pluggable model adapters**: MLX, OpenAI, OpenRouter, Exo cluster, Ollama
- **Hardware auto-detection**: Apple Silicon chip, memory, API keys, exo cluster, Ollama
- **SQLite vault** for responses, training entries, review sessions, sources
- **CI/CD** via GitHub Actions (Python tests on macOS, frontend build on Ubuntu)
- **786 tests**: 8 phase test files (280), functional tests (98), E2E tests (37), edge-case tests (89), round 2 feature tests (63), domain expansion tests (63), Xbox Blades tests (97)
- **Hermes skill** at ~/.hermes/skills/mlops/mindforge/

### Fixed

- Version mismatch: setup.py and __init__.py updated from 0.1.0 to 0.0.1
- RuntimeWarning: coroutine never awaited in server.py emit() function
- Format command: missing output directory creation (FileNotFoundError)
- CLI: KeyError/NameError risks (dict.get, import guards)
- CLI: improved error messages with actionable tips across all commands
- Frontend: tsc zero errors, build passes (403 modules)
- Tauri: cargo zero warnings, .app bundle valid

### Changed

- FastAPI: migrated from @app.on_event to lifespan context handler
- FastAPI: rewrote ingest-pdf and ingest-web endpoints with multi-step progress
- FastAPI: CORS scoped to specific localhost origins (was wildcard *)
- Design doc: updated TUI references to dashboard (Tauri GUI + CLI)
- Design doc: updated data directory from ~/mindforge-data/ to data/
