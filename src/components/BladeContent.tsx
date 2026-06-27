import { Children, type ReactNode, useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

const contentContainerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.03,
    },
  },
};

const contentChildVariants = {
  hidden: { y: 10, opacity: 0 },
  show: {
    y: 0,
    opacity: 1,
    transition: {
      duration: 0.2,
      ease: "easeOut",
    },
  },
};

const lineVariants = {
  hidden: { scaleX: 0, opacity: 0 },
  show: {
    scaleX: 1,
    opacity: 0.6,
    transition: {
      duration: 0.2,
      ease: "easeOut",
    },
  },
};

const reducedContentContainerVariants = {
  hidden: { opacity: 1 },
  show: { opacity: 1 },
};

const reducedContentChildVariants = {
  hidden: { y: 0, opacity: 1 },
  show: { y: 0, opacity: 1 },
};

const reducedLineVariants = {
  hidden: { scaleX: 1, opacity: 0.6 },
  show: { scaleX: 1, opacity: 0.6 },
};

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
  const [transitionState, setTransitionState] = useState<"gap" | "ready">(
    prefersReducedMotion ? "ready" : "gap",
  );
  const childNodes = Children.toArray(children);

  useEffect(() => {
    if (prefersReducedMotion) {
      setTransitionState("ready");
      return;
    }

    setTransitionState("gap");
    const timer = window.setTimeout(() => setTransitionState("ready"), 100);
    return () => window.clearTimeout(timer);
  }, [prefersReducedMotion, title]);

  const containerVariants = prefersReducedMotion
    ? reducedContentContainerVariants
    : contentContainerVariants;
  const childVariants = prefersReducedMotion
    ? reducedContentChildVariants
    : contentChildVariants;
  const headerLineVariants = prefersReducedMotion ? reducedLineVariants : lineVariants;

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
      <AnimatePresence initial={false}>
        {transitionState === "gap" && (
          <motion.div
            key="gap"
            initial={{ opacity: 1 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.1, ease: "easeOut" }}
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 5,
              background: "#000",
              pointerEvents: "none",
            }}
          />
        )}
      </AnimatePresence>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate={transitionState === "gap" ? "hidden" : "show"}
        style={{
          display: "flex",
          flexDirection: "column",
          height: "100%",
          minHeight: 0,
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Title header with gradient line */}
        <motion.div
          variants={childVariants}
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
            variants={headerLineVariants}
            style={{
              height: 1,
              background: "linear-gradient(90deg, var(--accent), transparent)",
              flex: 1,
              transformOrigin: "left center",
            }}
          />
        </motion.div>

        {/* Content area — fills full width with frosted glass, wireframe grid shows through */}
        <motion.div
          variants={childVariants}
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
          <motion.div
            variants={containerVariants}
            style={{ minHeight: "100%" }}
          >
            {childNodes.map((child, index) => (
              <motion.div
                key={index}
                variants={childVariants}
                style={{ minHeight: "100%" }}
              >
                {child}
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}
