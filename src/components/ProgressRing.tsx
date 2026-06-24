import { memo } from "react";
import { motion } from "framer-motion";

/**
 * Animated circular progress ring using SVG + framer-motion.
 * @param value - Progress percentage (0-100, clamped).
 * @param label - Optional label displayed below the percentage.
 * @param size - Diameter in pixels (default 120).
 */
export const ProgressRing = memo(function ProgressRing({ value, label, size = 120 }: { value: number; label?: string; size?: number }) {
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label ? `${label}: ${pct.toFixed(0)}%` : `Progress: ${pct.toFixed(0)}%`}
      style={{ width: size, height: size, position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size/2} cy={size/2} r={size/2 - 8} fill="none" stroke="var(--surface-raised)" strokeWidth={6} />
        <motion.circle
          cx={size/2} cy={size/2} r={size/2 - 8}
          fill="none" stroke="var(--accent)" strokeWidth={6}
          strokeDasharray={2 * Math.PI * (size/2 - 8)}
          initial={{ strokeDashoffset: 2 * Math.PI * (size/2 - 8) }}
          animate={{ strokeDashoffset: 2 * Math.PI * (size/2 - 8) * (1 - pct/100) }}
          style={{ filter: "drop-shadow(0 0 6px var(--accent-glow))" }}
        />
      </svg>
      <div style={{ position: "absolute", textAlign: "center" }}>
        <div style={{ fontSize: 24, fontWeight: 700, color: "var(--accent)" }}>{pct.toFixed(0)}%</div>
        {label && <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{label}</div>}
      </div>
    </div>
  );
});
