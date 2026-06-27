import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";
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
 * Xbox Blades-style VERTICAL blade bar on the right side of the screen.
 * Active blade expands with gold glow, angled edges, and a connected
 * vertical node column on the left edge of the menu.
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
  const activeIdx = BLADE_TABS.findIndex((t) => t.id === active);
  const prefersReducedMotion = useReducedMotion();

  return (
    <nav
      role="tablist"
      aria-label="Blade navigation"
      aria-orientation="vertical"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        alignItems: "stretch",
        gap: 2,
        padding: "12px 0",
        background: "linear-gradient(270deg, transparent 0%, rgba(0,0,0,0.4) 30%, var(--surface) 100%)",
        borderLeft: "1px solid var(--border)",
        position: "relative",
        zIndex: 20,
      }}
    >
      {/* Vertical gold accent line at top */}
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          bottom: 0,
          width: 1,
          background: "var(--accent)",
          boxShadow: "0 0 10px var(--accent), 0 0 18px var(--accent-glow)",
          opacity: 0.65,
          pointerEvents: "none",
        }}
      />

      {/* Vertical node connector column — circular nodes connected by glowing line */}
      <div
        className="vertical-node-connector"
        aria-hidden="true"
        style={{
          position: "absolute",
          left: 4,
          top: 20,
          bottom: 20,
          width: 12,
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-around",
          alignItems: "center",
          pointerEvents: "none",
        }}
      >
        {/* Thin glowing vertical line behind nodes */}
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: 0,
            bottom: 0,
            width: 1,
            background: "linear-gradient(180deg, transparent, var(--accent-dim), var(--accent-dim), transparent)",
            boxShadow: "0 0 6px var(--accent-glow)",
            opacity: 0.5,
            transform: "translateX(-50%)",
          }}
        />
        {BLADE_TABS.map((tab, i) => {
          const isActive = active === tab.id;
          return (
            <div
              key={tab.id}
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                position: "relative",
                zIndex: 1,
                background: isActive ? "var(--accent)" : "transparent",
                border: isActive
                  ? "1px solid var(--accent)"
                  : "1px solid var(--accent-dim)",
                boxShadow: isActive
                  ? "0 0 8px var(--accent), 0 0 14px var(--accent-glow)"
                  : "none",
                opacity: isActive ? 1 : 0.35,
                transition: "all 200ms ease",
              }}
            />
          );
        })}
      </div>

      {/* Vertical blade items */}
      {BLADE_TABS.map((tab, i) => {
        const isActive = active === tab.id;
        const isPrev = i === activeIdx - 1;
        const isNext = i === activeIdx + 1;

        return (
          <motion.button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            aria-label={tab.label}
            tabIndex={isActive ? 0 : -1}
            whileHover={prefersReducedMotion ? undefined : { x: -4, scale: 1.02 }}
            whileTap={prefersReducedMotion ? undefined : { x: -2, scale: 0.98 }}
            onMouseEnter={() => getSoundEngine().play("scroll")}
            onClick={() => {
              if (!isActive) {
                getSoundEngine().play("sweep");
                onSelect(tab.id);
              }
            }}
            animate={
              prefersReducedMotion
                ? { scaleX: 1, x: "0%" }
                : { scaleX: isActive ? 1 : 0.7, x: isActive ? "0%" : "10%" }
            }
            transition={prefersReducedMotion ? { duration: 0 } : { type: "spring", stiffness: 300, damping: 25 }}
            className={isActive ? "vertical-blade-item-active" : "vertical-blade-item"}
            style={{
              flex: 1,
              maxHeight: isActive ? 64 : 48,
              minHeight: 36,
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "flex-start",
              gap: 10,
              cursor: "pointer",
              border: "none",
              borderRadius: "8px 0 0 8px",
              padding: "8px 12px 8px 24px",
              position: "relative",
              // Frosted glass appearance
              background: isActive
                ? "linear-gradient(270deg, rgba(255,215,0,0.15) 0%, rgba(27,23,19,0.9) 60%, var(--surface-raised) 100%)"
                : "linear-gradient(270deg, rgba(54,48,41,0.4) 0%, rgba(27,23,19,0.6) 60%, transparent 100%)",
              backdropFilter: "blur(8px)",
              WebkitBackdropFilter: "blur(8px)",
              // Gold glow on active
              boxShadow: isActive
                ? "-6px 0 28px var(--accent-glow), 0 0 18px var(--accent-glow), inset 2px 0 0 var(--accent), inset 0 0 36px rgba(255,215,0,0.12)"
                : "inset 1px 0 0 rgba(205,127,50,0.2)",
              borderLeft: isActive
                ? "3px solid var(--accent)"
                : isPrev || isNext
                  ? "1px solid rgba(205,127,50,0.3)"
                  : "1px solid transparent",
              clipPath: "polygon(0 0, 100% 0, 100% 100%, 12px 100%)",
              color: isActive ? "var(--accent)" : "var(--text-dim)",
              fontWeight: isActive ? 600 : 400,
              fontSize: 12,
              opacity: isActive ? 1 : 0.4,
              transition: "color 150ms ease",
              transformOrigin: "right center",
              transformStyle: "preserve-3d",
            }}
          >
            {/* Icon */}
            <motion.span
              animate={{
                scale: isActive ? 1.3 : 1,
                opacity: 1,
              }}
              transition={{ duration: 0.2 }}
              style={{ display: "block", lineHeight: 1, fontSize: 18 }}
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
            {/* Active indicator bar — vertical on the left edge */}
            {isActive && (
              <motion.div
                layoutId="blade-active-bar"
                style={{
                  position: "absolute",
                  left: -1,
                  top: "20%",
                  bottom: "20%",
                  width: 3,
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
