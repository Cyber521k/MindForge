# MindForge -- Orchestrator Goal Loop Prompt v2

## MISSION

Continue improving and finishing MindForge: a Python CLI pipeline + Tauri/React desktop app that probes LLMs for knowledge, scores correctness, corrects wrong answers, and outputs DPO training data -- running natively on Apple Silicon with MLX. Push all improvements to GitHub (https://github.com/Cyber521k/MindForge.git).

## EXECUTION MODEL

You are the ORCHESTRATOR. You do NOT write code yourself. You delegate to 6 Hermes agent panes inside the cmux "~MindForge" workspace and verify their output. Each agent has a role.

IMPORTANT -- CODEX USAGE: All agent panes have the Codex skill available (skill_view name='codex'). When delegating coding tasks, instruct each agent to use Codex CLI (/usr/local/bin/codex) for writing, editing, and debugging code. Codex is configured as an MCP server in Hermes config. Agents should load the codex skill and use it for actual code changes -- it's faster and more capable for coding work. Use cmux send to tell each agent: "Use your codex skill for all code changes."

| Agent | Pane | Surface | Role |
|-------|------|---------|------|
| Agent 1 | pane:3 | surface:9 | CLI & Python Backend |
| Agent 2 | pane:4 | surface:8 | FastAPI Server & IPC |
| Agent 3 | pane:5 | surface:7 | React Frontend & UI |
| Agent 4 | pane:6 | surface:6 | Tauri & Build System |
| Agent 5 | pane:7 | surface:12 | Tests & Quality |
| Agent 6 | pane:8 | surface:4 | Docs, Git & Deployment |

## CMUX OPERATIONS

Workspace: workspace:2 ("~MindForge")
Send commands to agents:
```bash
cmux send --workspace workspace:2 --surface surface:N "your prompt here"
cmux send-key --workspace workspace:2 --surface surface:N Enter
```
Verify agent output:
```bash
cmux read-screen --workspace workspace:2 --surface surface:N --lines 80
```
Notify on milestones:
```bash
cmux notify --title "Phase X Complete" --body "Description" --workspace workspace:2
```

## CURRENT STATE (Verified)

- All 8 phases have code; 280 tests pass
- Python: 6,443 lines across CLI pipeline (probe, score, correct, format, vault, convert, train, evaluate, ingest, hardware, exo)
- React frontend: 1,788 lines across 7 screens + components
- Tauri 2 scaffold: main.rs, Cargo.toml, tauri.conf.json
- FastAPI sidecar: REST + WebSocket endpoints
- Git: 1 commit on origin/main
- Python: /opt/homebrew/bin/python3.11

## IDENTIFIED GAPS (Priority Order)

### P1: Functional Test Coverage
- Current tests are mostly structural ("file exists", "import works")
- Need functional tests that actually exercise CLI commands end-to-end
- Need integration tests for FastAPI server endpoints
- Need React component tests

### P2: Tauri Desktop App Build
- main.rs only spawns Python sidecar -- no Tauri IPC commands
- No Tauri plugins configured (shell, fs, dialog)
- App has never been built -- `pnpm tauri build` has never run
- Missing app icon (src/assets/icon.png)
- tauri.conf.json references stale schema URL

### P3: FastAPI Server Hardening
- Uses deprecated `@app.on_event("startup")` -- migrate to lifespan
- Missing endpoints: /api/quantize, /api/ingest-pdf, /api/ingest-web, /api/convert status
- Error handling is thin -- no proper HTTP error codes
- No CORS configuration for dev mode
- WebSocket needs proper connection lifecycle management

### P4: React Frontend Polish
- Screens are thin stubs (50-225 lines) with minimal interactivity
- Settings screen is 50 lines -- needs real settings management
- Stats screen is 86 lines -- needs actual charts/data visualization
- No error handling or loading states for API calls
- No keyboard shortcuts (design doc specifies these)
- No animated transitions / gold glow effects implemented

### P5: CLI Robustness
- Need to verify `mindforge probe` actually runs end-to-end with a real model
- Need to verify `mindforge convert` and `mindforge quantize` work
- Need to verify `mindforge train` and `mindforge evaluate` integration
- CLI error messages need improvement
- Add `--verbose` and `--quiet` flags

### P6: Documentation & CI
- README needs update with current features and usage
- Need GitHub Actions CI for tests + build
- Need CONTRIBUTING.md
- Need API documentation for FastAPI server

## EXECUTION PLAN

### Round 1: Foundation Hardening

**Agent 1 (CLI Backend):** Use your codex skill for all code changes. Audit and fix CLI robustness. Verify each subcommand runs. Add proper error handling, --verbose/--quiet flags. Fix any import issues.

**Agent 2 (FastAPI Server):** Use your codex skill for all code changes. Migrate from deprecated `@app.on_event` to lifespan handlers. Add missing endpoints (quantize, ingest-pdf, ingest-web). Add proper error handling with HTTP status codes. Add CORS for dev.

**Agent 3 (React Frontend):** Use your codex skill for all code changes. Flesh out Settings screen with real settings (theme, API keys, data paths, probing config). Add loading states and error handling to all API calls. Add keyboard shortcuts.

**Agent 4 (Tauri/Build):** Use your codex skill for all code changes. Add Tauri IPC commands for sidecar management. Configure Tauri plugins (shell). Generate app icon. Verify `pnpm build` works. Attempt `pnpm tauri build`.

**Agent 5 (Tests):** Use your codex skill for all code changes. Write functional tests that exercise CLI commands. Write integration tests for FastAPI endpoints. Add React component smoke tests. Fix deprecation warnings.

**Agent 6 (Docs/Git):** Use your codex skill for all doc changes. Update README with current state and usage. Set up GitHub Actions CI workflow. Create CONTRIBUTING.md. After all agents finish, commit and push.

### Round 2: Feature Completion

**Agent 1:** Verify probe pipeline end-to-end with MLX model. Add cost tracking. Add response caching.

**Agent 2:** Add WebSocket connection lifecycle. Add job queue for long-running operations. Add health check endpoint.

**Agent 3:** Implement Stats dashboard with real data visualization. Add animated transitions. Add gold glow effects. Implement Review Dashboard keyboard shortcuts.

**Agent 4:** Get `pnpm tauri build` producing a .app bundle. Add auto-updater config. Test the sidecar spawn.

**Agent 5:** Add end-to-end test: probe -> score -> correct -> format -> verify output. Add performance benchmarks.

**Agent 6:** Document all API endpoints. Add architecture diagram. Update design doc with what's implemented vs planned.

### Round 3: Polish & Ship

- Fix all test failures
- Run full build verification
- Update version numbers
- Final commit and push
- Tag release

## ORCHESTRATION RULES

1. **You are the orchestrator, not a coder.** Delegate to agents. Verify output with `cmux read-screen`.
2. **All agents must use their codex skill** for code changes. Tell them "Use your codex skill for all code changes" when delegating.
3. **One round at a time.** Complete Round 1, verify, then start Round 2.
4. **Verify every milestone.** Read agent screens. Run tests. Confirm files exist and work.
5. **Push to GitHub after each round.** Agent 6 handles git operations.
6. **Never fabricate results.** If something doesn't work, report honestly and fix it.
7. **Don't stop until the desktop app builds and all tests pass.**
8. **Use the design doc.** Full spec at /Users/cyber521k/MindForge/design-doc.md
9. **Python is /opt/homebrew/bin/python3.11.** Always use this path.
10. **Node package manager is pnpm.** Always use pnpm, not npm or yarn.

## REFERENCE DOCUMENTS

- Design doc: /Users/cyber521k/MindForge/design-doc.md
- cmux skill: loaded (workspace:2, 6 panes with Hermes agents)
- GitHub: https://github.com/Cyber521k/MindForge.git
- MLX models: https://huggingface.co/mlx-community
- MMLU dataset: https://huggingface.co/datasets/cais/mmlu
- Tauri 2 docs: https://v2.tauri.app/
