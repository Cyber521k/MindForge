import type { Screen } from "../lib/theme";

const PHASE_LABELS: Record<Screen, string> = {
  "model-setup": "Model Setup",
  "domain-setup": "Domain Setup",
  probing: "Probing",
  review: "Review",
  format: "Format & Export",
  train: "Train & Evaluate",
  stats: "Statistics",
  settings: "Settings",
};

/**
 * Bottom status bar showing model, phase, progress, and WebSocket connection state.
 * @param model - Optional selected model name.
 * @param phase - Optional current screen/phase ID.
 * @param progress - Optional progress text to display.
 * @param connected - WebSocket connection status (green dot if true).
 */
export function StatusBar({
  model,
  phase,
  progress,
  connected,
}: {
  model?: string;
  phase?: Screen | string;
  progress?: string;
  connected?: boolean;
}) {
  return (
    <div
      style={{
        height: 32,
        background: "var(--surface)",
        borderTop: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        padding: "0 16px",
        gap: 16,
        fontSize: 12,
        color: "var(--text-secondary)",
        flexShrink: 0,
      }}
    >
      <span style={{ color: "var(--accent)", fontSize: 14 }} aria-hidden="true">⚕</span>
      {model && <span>[Model: {model}]</span>}
      {phase && <span>[Phase: {typeof phase === "string" ? PHASE_LABELS[phase as Screen] || phase : phase}]</span>}
      {progress && <span>[{progress}]</span>}
      <span style={{ marginLeft: "auto", color: connected ? "var(--success)" : "var(--error)" }}>
        {connected ? "● Connected" : "○ Disconnected"}
      </span>
      <span style={{ color: "var(--text-dim)" }}>MindForge v7.0</span>
    </div>
  );
}
