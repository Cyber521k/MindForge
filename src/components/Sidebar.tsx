import { memo } from "react";
import { motion } from "framer-motion";
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
 * Xbox Blades-style navigation sidebar.
 * Each nav item is a vertical blade tab. Active blade has gold border, glow,
 * and a clipped right edge that points toward the content area.
 */
export const Sidebar = memo(function Sidebar({
  active,
  onSelect,
  model,
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
        width: 200,
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
        position: "relative",
        zIndex: 10,
        transformStyle: "preserve-3d",
      }}
    >
      {/* Header */}
      <div style={{ padding: "18px 16px", display: "flex", alignItems: "center", gap: 10 }}>
        <Caduceus size={28} />
        <span style={{ fontSize: 17, fontWeight: 700, color: "var(--accent)" }}>MindForge</span>
      </div>

      {/* Blade tabs */}
      <div style={{ flex: 1, padding: "4px 0", overflowY: "auto", position: "relative" }}>
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          return (
            <motion.div
              key={item.id}
              role="button"
              tabIndex={0}
              aria-current={isActive ? "page" : undefined}
              aria-label={item.label}
              whileHover={{ x: 6 }}
              whileTap={{ x: 2 }}
              onClick={() => onSelect(item.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(item.id);
                }
              }}
              style={{
                padding: "11px 16px 11px 14px",
                marginBottom: 2,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 10,
                fontSize: 13,
                fontWeight: isActive ? 600 : 400,
                color: isActive ? "var(--accent)" : "var(--text)",
                background: isActive
                  ? "linear-gradient(90deg, var(--surface-raised) 0%, var(--surface) 85%, transparent 100%)"
                  : "transparent",
                borderLeft: isActive
                  ? "3px solid var(--accent)"
                  : "3px solid transparent",
                boxShadow: isActive
                  ? "inset 0 0 20px var(--accent-glow), -2px 0 8px var(--accent-glow)"
                  : "none",
                // Clip-path creates the angled blade edge on the right
                clipPath: isActive
                  ? "polygon(0 0, calc(100% - 12px) 0, 100% 50%, calc(100% - 12px) 100%, 0 100%)"
                  : "polygon(0 0, 100% 0, 100% 100%, 0 100%)",
                paddingRight: isActive ? 24 : 16,
                transition: "background 150ms ease, color 150ms ease, box-shadow 150ms ease",
                position: "relative",
              }}
            >
              <span style={{ fontSize: 15 }} aria-hidden="true">{item.icon}</span>
              <span style={{
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}>
                {item.label}
              </span>
              {/* Gold accent line on right edge of active blade */}
              {isActive && (
                <motion.div
                  layoutId="blade-indicator"
                  style={{
                    position: "absolute",
                    right: 0,
                    top: "10%",
                    bottom: "10%",
                    width: 2,
                    background: "var(--accent)",
                    borderRadius: 2,
                    boxShadow: "0 0 8px var(--accent)",
                  }}
                />
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "10px 16px",
          borderTop: "1px solid var(--border)",
          fontSize: 10,
          color: "var(--text-dim)",
        }}
      >
        <div>v0.0.1</div>
        {model && (
          <div style={{ color: "var(--text-secondary)", fontSize: 10, marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            ⚕ {model.split("/").pop()}
          </div>
        )}
      </div>
    </nav>
  );
});
