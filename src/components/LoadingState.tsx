import { motion } from "framer-motion";

/**
 * Reusable loading state with pulsing caduceus icon.
 */
export function LoadingState({ message = "Loading..." }: { message?: string }) {
  return (
    <div style={{ padding: 40, color: "var(--text-secondary)", textAlign: "center" }}>
      <motion.div
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        style={{ fontSize: 32, marginBottom: 12, color: "var(--accent)" }}
      >
        ⚕
      </motion.div>
      <motion.div
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        style={{ fontSize: 14 }}
      >
        {message}
      </motion.div>
    </div>
  );
}
