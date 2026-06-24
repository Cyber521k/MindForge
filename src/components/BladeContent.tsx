import { type ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";

/**
 * BladeContent — wraps screen content in the Xbox blade layout.
 * Left section: large decorative icon (35% width).
 * Right section: scrollable content area (65% width).
 * Frosted glass panel with gold edge glow.
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
        height: "100%",
        gap: 0,
        overflow: "hidden",
        position: "relative",
        // Radial spotlight gradient
        background:
          "radial-gradient(ellipse 80% 60% at 30% 40%, rgba(255,215,0,0.04) 0%, transparent 60%), var(--bg)",
      }}
    >
      {/* Left: decorative icon panel */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1, duration: 0.3 }}
        style={{
          width: "32%",
          minWidth: 180,
          maxWidth: 300,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: 24,
          position: "relative",
          // Frosted glass
          background:
            "linear-gradient(135deg, rgba(54,48,41,0.3) 0%, rgba(27,23,19,0.5) 100%)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          borderRight: "1px solid rgba(205,127,50,0.2)",
          boxShadow: "inset -4px 0 20px rgba(0,0,0,0.3)",
          overflow: "hidden",
        }}
      >
        {/* Large decorative icon with glow — opacity-only animation (compositor-friendly) */}
        <motion.div
          animate={prefersReducedMotion ? { opacity: 1 } : { opacity: [0.7, 1, 0.7] }}
          transition={prefersReducedMotion ? { duration: 0 } : {
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            fontSize: 96,
            lineHeight: 1,
            color: "var(--accent)",
            textShadow:
              "0 0 30px var(--accent-glow), 0 0 60px var(--accent-glow), 0 0 90px rgba(255,215,0,0.1)",
            marginBottom: 16,
            willChange: "opacity",
          }}
          aria-hidden="true"
        >
          {icon}
        </motion.div>

        {/* Title */}
        <div
          style={{
            fontSize: 18,
            fontWeight: 700,
            color: "var(--accent)",
            textTransform: "uppercase",
            letterSpacing: 2,
            textAlign: "center",
            textShadow: "0 0 10px var(--accent-glow)",
          }}
        >
          {title}
        </div>

        {/* Decorative line */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "60%" }}
          transition={{ delay: 0.3, duration: 0.4 }}
          style={{
            height: 2,
            background: "linear-gradient(90deg, transparent, var(--accent), transparent)",
            marginTop: 12,
            opacity: 0.6,
          }}
        />
      </motion.div>

      {/* Right: content area */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15, duration: 0.3 }}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 24,
          position: "relative",
        }}
      >
        {children}
      </motion.div>
    </div>
  );
}
