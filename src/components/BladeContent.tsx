import { type ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";

/**
 * BladeContent — wraps screen content in the Xbox blade layout.
 * The orb/icon is now in App.tsx, not here.
 * Content fills the full area with frosted glass and radial spotlight.
 * Keeps the title header with gradient line and entrance animations.
 * The content area shows the wireframe grid background through it.
 */
export function BladeContent({
  icon,
  title,
  children,
}: {
  icon: string;
  title: string;
  children: ReactNode;
}) {
  const prefersReducedMotion = useReducedMotion();
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
        position: "relative",
        // Radial spotlight gradient
        background:
          "radial-gradient(ellipse 80% 60% at 30% 40%, rgba(255,215,0,0.04) 0%, transparent 60%), var(--bg)",
      }}
    >
      {/* Title header with gradient line */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05, duration: 0.25 }}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "16px 32px 12px",
          flexShrink: 0,
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Icon in the header (small, not the large left panel) */}
        <span
          style={{
            fontSize: 20,
            color: "var(--accent)",
            textShadow: "0 0 12px var(--accent-glow)",
          }}
          aria-hidden="true"
        >
          {icon}
        </span>
        {/* Title text */}
        <div
          style={{
            fontSize: 18,
            fontWeight: 700,
            color: "var(--accent)",
            textTransform: "uppercase",
            letterSpacing: 3,
            textShadow: "0 0 10px var(--accent-glow)",
          }}
        >
          {title}
        </div>
        {/* Gradient line filling remaining width */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          transition={{ delay: 0.2, duration: 0.4 }}
          style={{
            height: 1,
            background: "linear-gradient(90deg, var(--accent), transparent)",
            opacity: 0.6,
            flex: 1,
          }}
        />
      </motion.div>

      {/* Content area — fills full width with frosted glass, wireframe grid shows through */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.3 }}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 32,
          position: "relative",
          zIndex: 1,
          // Frosted glass — wireframe grid shows through
          background: "rgba(27, 23, 19, 0.35)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
        }}
      >
        {children}
      </motion.div>
    </div>
  );
}
