import { motion } from "framer-motion";

/**
 * Reusable error state with optional retry button.
 * Shows error icon, message, and retry button.
 */
export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel"
      style={{ padding: 24, textAlign: "center" }}
    >
      <div style={{ fontSize: 36, color: "var(--error)", marginBottom: 12 }}>✗</div>
      <div style={{ color: "var(--error)", fontSize: 15, fontWeight: 600, marginBottom: 8 }}>
        {message}
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="btn-gold"
          style={{ marginTop: 4, padding: "8px 20px", fontSize: 14 }}
        >
          ↻ Retry
        </button>
      )}
    </motion.div>
  );
}
