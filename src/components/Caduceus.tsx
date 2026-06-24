import { motion } from "framer-motion";

/**
 * Hermes caduceus (⚕) logo component — animated gold glow.
 */
export function Caduceus({ size = 48, animate = true }: { size?: number; animate?: boolean }) {
  if (!animate) {
    return (
      <div style={{ fontSize: size, lineHeight: 1 }} className="caduceus">
        ⚕
      </div>
    );
  }
  return (
    <motion.div
      animate={{ opacity: [0.8, 1, 0.8] }}
      transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
      style={{ fontSize: size, lineHeight: 1 }}
      className="caduceus"
    >
      ⚕
    </motion.div>
  );
}
