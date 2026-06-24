import { memo } from "react";
import { motion } from "framer-motion";
import type { Screen } from "../lib/theme";
import { getSoundEngine } from "./SoundManager";

const BLADE_TABS: { id: Screen; label: string; icon: string }[] = [
  { id: "model-setup", label: "Model", icon: "🖥" },
  { id: "domain-setup", label: "Domains", icon: "📚" },
  { id: "probing", label: "Probe", icon: "🔍" },
  { id: "review", label: "Review", icon: "📋" },
  { id: "format", label: "Format", icon: "📦" },
  { id: "train", label: "Train", icon: "🎯" },
  { id: "stats", label: "Stats", icon: "📊" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

/**
 * Xbox Blades-style horizontal blade bar at the bottom of the screen.
 * Active blade expands upward with gold glow and angled edges.
 */
export const BladeBar = memo(function BladeBar({
  active,
  onSelect,
  direction,
}: {
  active: Screen;
  onSelect: (s: Screen) => void;
  direction: number;
}) {
  return (
    <nav
      aria-label="Blade navigation"
      style={{
        display: "flex",
        height: "100%",
        alignItems: "stretch",
        gap: 2,
        padding: "0 8px",
        background: "linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.4) 30%, var(--surface) 100%)",
        borderTop: "1px solid var(--border)",
        position: "relative",
        zIndex: 20,
      }}
    >
      {BLADE_TABS.map((tab, i) => {
        const isActive = active === tab.id;
        const isPrev = i === BLADE_TABS.findIndex((t) => t.id === active) - 1;
        const isNext = i === BLADE_TABS.findIndex((t) => t.id === active) + 1;

        return (
          <motion.button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            aria-label={tab.label}
            tabIndex={isActive ? 0 : -1}
            whileHover={{ y: -4, scale: 1.03 }}
            whileTap={{ y: -2, scale: 0.98 }}
            onClick={() => {
              if (!isActive) {
                getSoundEngine().play("sweep");
                onSelect(tab.id);
              }
            }}
            animate={{
              height: isActive ? "100%" : "70%",
              marginTop: isActive ? 0 : "auto",
              marginBottom: 0,
            }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            style={{
              flex: 1,
              maxWidth: isActive ? 180 : 120,
              minWidth: 80,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
              cursor: "pointer",
              border: "none",
              borderRadius: "8px 8px 0 0",
              padding: "8px 4px",
              position: "relative",
              // Frosted glass appearance
              background: isActive
                ? "linear-gradient(180deg, rgba(255,215,0,0.15) 0%, rgba(27,23,19,0.9) 60%, var(--surface-raised) 100%)"
                : "linear-gradient(180deg, rgba(54,48,41,0.4) 0%, rgba(27,23,19,0.6) 60%, transparent 100%)",
              backdropFilter: "blur(8px)",
              WebkitBackdropFilter: "blur(8px)",
              // Gold glow on active
              boxShadow: isActive
                ? "0 -4px 20px var(--accent-glow), inset 0 1px 0 var(--accent), inset 0 0 30px rgba(255,215,0,0.08)"
                : "inset 0 1px 0 rgba(205,127,50,0.2)",
              borderTop: isActive
                ? "2px solid var(--accent)"
                : isPrev || isNext
                  ? "1px solid rgba(205,127,50,0.3)"
                  : "1px solid transparent",
              clipPath: isActive
                ? "polygon(8px 0, calc(100% - 8px) 0, 100% 100%, 0 100%)"
                : "polygon(4px 0, calc(100% - 4px) 0, 100% 100%, 0 100%)",
              color: isActive ? "var(--accent)" : "var(--text-dim)",
              fontWeight: isActive ? 600 : 400,
              fontSize: 12,
              transition: "color 150ms ease",
              transformStyle: "preserve-3d",
            }}
          >
            {/* Large icon */}
            <motion.span
              animate={{
                fontSize: isActive ? 28 : 20,
                opacity: isActive ? 1 : 0.5,
              }}
              transition={{ duration: 0.2 }}
              style={{ display: "block", lineHeight: 1 }}
              aria-hidden="true"
            >
              {tab.icon}
            </motion.span>
            {/* Label */}
            <span style={{
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              maxWidth: "100%",
              textShadow: isActive ? "0 0 8px var(--accent-glow)" : "none",
            }}>
              {tab.label}
            </span>
            {/* Active indicator bar */}
            {isActive && (
              <motion.div
                layoutId="blade-active-bar"
                style={{
                  position: "absolute",
                  top: -1,
                  left: "20%",
                  right: "20%",
                  height: 3,
                  background: "var(--accent)",
                  borderRadius: 2,
                  boxShadow: "0 0 12px var(--accent)",
                }}
              />
            )}
          </motion.button>
        );
      })}
    </nav>
  );
});
