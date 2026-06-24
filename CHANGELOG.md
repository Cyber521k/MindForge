# Changelog

All notable changes to MindForge are documented in this file.

## [7.0.2] - 2026-06-24 (Round 3)

### Added

- **Xbox Blades navigation system**: Complete UI redesign inspired by the original Xbox dashboard
  - 3D blade sweep transitions (Framer Motion `rotateY: 45deg`, `translateX`, direction-aware)
  - CSS `perspective: 1400px` container with `preserve-3d` for depth
  - Blade tabs with angled right edges (CSS `clip-path` polygon), gold gradient, inset glow
  - Gold accent indicator bar on active blade (Framer Motion `layoutId`)
  - `aria-live="polite"` for screen reader announcements on blade change
  - `AnimatePresence mode="wait"` to prevent focus trapping
- **Arrow key navigation**: ArrowLeft/ArrowRight switch between blades, ArrowUp/ArrowDown for in-screen content
- **Direction-aware transitions**: Navigation direction determines sweep direction (forward = right-to-left, back = left-to-right)
- **Blade CSS classes**: `.blade-container`, `.blade-panel`, `.blade-tab-active`, `.blade-glow` in index.css
- **Sidebar as memo**: `React.memo` with angled clip-path tabs and gold indicator
- **Round 2 feature tests**: 63 new tests (test_round2_features.py) -- Ollama integration, async endpoints, DB indexes, ErrorBoundary

### Planned (Not Yet Implemented)

- **Sound effects**: Whoosh on blade change, click on select, tick on scroll, ambient music (Web Audio API / Howler.js)
- **Controller hints**: A=Select, B=Back at bottom corners
- **Frosted glass / translucent blade appearance** with glowing edges
- **Hexagonal grid background pattern** with radial spotlight gradient

### Changed

- Test count: 504 -> 567 (63 new round 2 feature tests)
- Test files: 11 -> 12 (added test_round2_features.py)
- Screen transitions: simple slide -> 3D blade sweep with perspective and rotateY
- Navigation: setScreen -> navigate() with direction tracking
- Sidebar: vertical list -> blade tabs with clip-path angled edges and gold glow

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

## [7.0.0] - 2026-06-24

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
- **567 tests**: 8 phase test files (280), functional tests (98), E2E tests (37), edge-case tests (89), round 2 feature tests (63)
- **Hermes skill** at ~/.hermes/skills/mlops/mindforge/

### Fixed

- Version mismatch (setup.py and __init__.py: 0.1.0 -> 7.0.0)
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
