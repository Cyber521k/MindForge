import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import type { Screen } from "../lib/theme";
import { getSoundEngine } from "./SoundManager";

const XBOX_CHARTREUSE = "var(--xbox-chartreuse, #CCFF00)";
const XBOX_DIM_GREEN = "var(--xbox-dim-green, #1A3A1A)";
const XBOX_GLOW = "var(--xbox-glow, rgba(204, 255, 0, 0.58))";

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

const bladeEntranceVariants = {
  hidden: { x: 100, opacity: 0 },
  visible: (index: number) => ({
    x: 0,
    opacity: 1,
    transition: {
      delay: index * 0.05,
      duration: 0.3,
      ease: "easeOut",
    },
  }),
};

function playHoverSound() {
  const soundEngine = getSoundEngine();
  if (!soundEngine.isMuted()) {
    soundEngine.play("scroll");
  }
}

/**
 * Xbox Blades-style VERTICAL blade bar on the right side of the screen.
 * Active blade glows with chartreuse edges, angled tabs, and a connected
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
          background: XBOX_CHARTREUSE,
          boxShadow: `0 0 10px ${XBOX_CHARTREUSE}, 0 0 18px ${XBOX_GLOW}`,
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
            background: `linear-gradient(180deg, transparent, ${XBOX_DIM_GREEN}, ${XBOX_DIM_GREEN}, transparent)`,
            boxShadow: `0 0 6px ${XBOX_GLOW}`,
            opacity: 0.5,
            transform: "translateX(-50%)",
          }}
        />
        {BLADE_TABS.map((tab) => {
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
                background: isActive ? XBOX_CHARTREUSE : "transparent",
                border: isActive
                  ? `2px solid ${XBOX_CHARTREUSE}`
                  : `1px solid ${XBOX_DIM_GREEN}`,
                boxShadow: isActive
                  ? `0 0 8px ${XBOX_CHARTREUSE}, 0 0 14px ${XBOX_GLOW}`
                  : `0 0 4px ${XBOX_DIM_GREEN}`,
                opacity: isActive ? 1 : 0.35,
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
          <motion.div
            key={tab.id}
            role="presentation"
            custom={i}
            variants={prefersReducedMotion ? undefined : bladeEntranceVariants}
            initial={prefersReducedMotion ? false : "hidden"}
            animate={prefersReducedMotion ? undefined : "visible"}
            style={{
              flex: 1,
              minHeight: 36,
              display: "flex",
              transformOrigin: "right center",
            }}
          >
            <motion.button
              role="tab"
              aria-selected={isActive}
              aria-label={tab.label}
              tabIndex={isActive ? 0 : -1}
              whileHover={prefersReducedMotion ? undefined : { x: -4, scale: 1.02 }}
              whileTap={prefersReducedMotion ? undefined : { x: -2, scale: 0.98 }}
              onMouseEnter={playHoverSound}
              onClick={() => {
                if (!isActive) {
                  getSoundEngine().play("sweep");
                  onSelect(tab.id);
                }
              }}
              animate={
                prefersReducedMotion
                  ? {
                      scaleX: 1,
                      borderColor: isActive ? "#CCFF00" : "#1A3A1A",
                    }
                  : {
                      scaleX: isActive ? 1 : 0.72,
                      borderColor: isActive
                        ? ["#CCFF00", "rgba(51,255,51,0.6)"]
                        : "rgba(26,58,26,0.85)",
                    }
              }
              transition={
                prefersReducedMotion
                  ? { duration: 0 }
                  : {
                      scaleX: { type: "spring", stiffness: 300, damping: 25 },
                      borderColor: { duration: 0.2, ease: "easeOut" },
                    }
              }
              className={isActive ? undefined : "vertical-blade-item"}
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "flex-start",
                gap: 10,
                cursor: "pointer",
                border: isActive ? `2px solid ${XBOX_CHARTREUSE}` : `1px solid ${XBOX_DIM_GREEN}`,
                borderRadius: "8px 0 0 8px",
                padding: "8px 12px 8px 24px",
                position: "relative",
                overflow: "hidden",
                // Frosted glass appearance
                background: isActive
                  ? "linear-gradient(270deg, rgba(204,255,0,0.18) 0%, rgba(13,42,13,0.92) 58%, rgba(4,28,28,0.88) 100%)"
                  : "linear-gradient(270deg, rgba(26,58,26,0.34) 0%, rgba(13,42,13,0.45) 58%, transparent 100%)",
                backdropFilter: "blur(8px)",
                WebkitBackdropFilter: "blur(8px)",
                // Chartreuse glow on active
                boxShadow: isActive
                  ? `-8px 0 30px ${XBOX_GLOW}, 0 0 20px ${XBOX_GLOW}, inset 3px 0 0 ${XBOX_CHARTREUSE}, inset 0 0 36px rgba(204,255,0,0.12)`
                  : `-2px 0 8px rgba(26,58,26,0.35), inset 1px 0 0 ${XBOX_DIM_GREEN}`,
                borderLeft: isActive
                  ? `2px solid ${XBOX_CHARTREUSE}`
                  : isPrev || isNext
                    ? `1px solid ${XBOX_DIM_GREEN}`
                    : "1px solid rgba(26,58,26,0.45)",
                clipPath: "polygon(0 0, 100% 0, 100% 100%, 12px 100%)",
                color: isActive ? XBOX_CHARTREUSE : "var(--text-dim)",
                fontWeight: isActive ? 600 : 400,
                fontSize: 12,
                opacity: isActive ? 1 : 0.48,
                transition: "none",
                transformOrigin: "right center",
                transformStyle: "preserve-3d",
              }}
            >
              {isActive && (
                <motion.div
                  aria-hidden="true"
                  initial={{ opacity: 0 }}
                  animate={prefersReducedMotion ? { opacity: 0 } : { opacity: [1, 0] }}
                  transition={{ duration: 0.2, ease: "easeOut" }}
                  style={{
                    position: "absolute",
                    inset: 0,
                    border: "2px solid #CCFF00",
                    boxShadow: "0 0 18px #CCFF00, inset 0 0 18px rgba(204,255,0,0.35)",
                    pointerEvents: "none",
                    clipPath: "polygon(0 0, 100% 0, 100% 100%, 12px 100%)",
                  }}
                />
              )}
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
              <span
                style={{
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  maxWidth: "100%",
                  textShadow: isActive ? `0 0 8px ${XBOX_GLOW}` : "none",
                }}
              >
                {tab.label}
              </span>
              {isActive && (
                <motion.span
                  className="xbox-cursor-bracket"
                  aria-hidden="true"
                  initial={{ opacity: 0, scaleX: 0.85 }}
                  animate={{ opacity: 1, scaleX: 1 }}
                  transition={prefersReducedMotion ? { duration: 0 } : { duration: 0.16, ease: "easeOut" }}
                  style={{
                    marginLeft: "auto",
                    color: XBOX_CHARTREUSE,
                    fontWeight: 700,
                    lineHeight: 1,
                    textShadow: `0 0 10px ${XBOX_GLOW}`,
                    transformOrigin: "center",
                  }}
                >
                  [ ]
                </motion.span>
              )}
              {/* Active indicator bar — vertical on the left edge */}
              {isActive && (
                <motion.div
                  initial={{ opacity: 0, scaleY: 0.7 }}
                  animate={{ opacity: 1, scaleY: 1 }}
                  transition={prefersReducedMotion ? { duration: 0 } : { duration: 0.16, ease: "easeOut" }}
                  style={{
                    position: "absolute",
                    left: -1,
                    top: "20%",
                    bottom: "20%",
                    width: 3,
                    background: XBOX_CHARTREUSE,
                    borderRadius: 2,
                    boxShadow: `0 0 12px ${XBOX_CHARTREUSE}, 0 0 20px ${XBOX_GLOW}`,
                    transformOrigin: "center",
                  }}
                />
              )}
            </motion.button>
          </motion.div>
        );
      })}
    </nav>
  );
});
