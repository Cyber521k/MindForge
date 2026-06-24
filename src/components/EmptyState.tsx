import { motion } from "framer-motion";

/**
 * Empty state component for when a screen has no data to display.
 * Shows an icon, message, and optional action hint.
 */
export function EmptyState({
  icon = "📭",
  title = "No data yet",
  message,
  action,
}: {
  icon?: string;
  title?: string;
  message?: string;
  action?: React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel"
      style={{ padding: 40, textAlign: "center" }}
    >
      <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.6 }}>{icon}</div>
      <div style={{ color: "var(--text-secondary)", fontSize: 16, marginBottom: 8 }}>{title}</div>
      {message && (
        <div style={{ color: "var(--text-dim)", fontSize: 13, marginBottom: 20 }}>{message}</div>
      )}
      {action}
    </motion.div>
  );
}
