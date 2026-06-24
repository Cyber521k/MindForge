import { memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Caduceus } from "./Caduceus";
import type { Screen } from "../lib/theme";

const NAV_ITEMS: { id: Screen; label: string; icon: string }[] = [
  { id: "model-setup", label: "Model Setup", icon: "🖥" },
  { id: "domain-setup", label: "Domain Setup", icon: "📚" },
  { id: "probing", label: "Probe Engine", icon: "🔍" },
  { id: "review", label: "Score & Review", icon: "✓" },
  { id: "format", label: "Format & Export", icon: "📦" },
  { id: "train", label: "Train & Evaluate", icon: "🎯" },
  { id: "stats", label: "Stats", icon: "📊" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

/**
 * Navigation sidebar with 8 screen tabs and version footer.
 * @param active - Currently active screen ID.
 * @param onSelect - Callback when a nav item is clicked.
 * @param model - Optional selected model name (shown in footer).
 * @param phase - Optional current pipeline phase (shown in footer).
 */
export const Sidebar = memo(function Sidebar({
  active,
  onSelect,
  model,
  phase,
}: {
  active: Screen;
  onSelect: (s: Screen) => void;
  model?: string;
  phase?: string;
}) {
  return (
    <nav
      aria-label="Main navigation"
      style={{
        width: 220,
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div style={{ padding: "20px 16px", display: "flex", alignItems: "center", gap: 12 }}>
        <Caduceus size={32} />
        <span style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>MindForge</span>
      </div>

      {/* Navigation */}
      <div style={{ flex: 1, padding: "8px 8px", overflowY: "auto" }}>
        {NAV_ITEMS.map((item, i) => (
          <motion.div
            key={item.id}
            role="button"
            tabIndex={0}
            aria-current={active === item.id ? "page" : undefined}
            aria-label={item.label}
            whileHover={{ x: 4 }}
            onClick={() => onSelect(item.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(item.id);
              }
            }}
            style={{
              padding: "10px 16px",
              marginBottom: 2,
              borderRadius: 6,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: 14,
              color: active === item.id ? "var(--accent)" : "var(--text)",
              background: active === item.id ? "var(--surface-raised)" : "transparent",
              borderLeft:
                active === item.id ? "3px solid var(--accent)" : "3px solid transparent",
              transition: "background 100ms ease, color 100ms ease",
            }}
          >
            <span style={{ fontSize: 16 }} aria-hidden="true">{item.icon}</span>
            {item.label}
          </motion.div>
        ))}
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "12px 16px",
          borderTop: "1px solid var(--border)",
          fontSize: 11,
          color: "var(--text-dim)",
        }}
      >
        <div>v7.0.0</div>
        {model && <div style={{ color: "var(--text-secondary)" }}>⚕ {model.split("/").pop()}</div>}
      </div>
    </nav>
  );
});
