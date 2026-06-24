import { motion } from "framer-motion";

/**
 * Skeleton loading card — animated placeholder for panels while data loads.
 */
export function SkeletonCard({ lines = 3 }: { lines?: number }) {
  return (
    <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
      {Array.from({ length: lines }).map((_, i) => (
        <motion.div
          key={i}
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.1, ease: "easeInOut" }}
          style={{
            height: 14,
            borderRadius: 4,
            background: "var(--surface-raised)",
            marginBottom: 12,
            width: `${70 + Math.sin(i * 1.7) * 25}%`,
          }}
        />
      ))}
    </div>
  );
}
