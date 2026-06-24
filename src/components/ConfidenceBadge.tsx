import { memo } from "react";

/**
 * Confidence badge showing auto/review/low status with color coding.
 * @param confidence - Confidence score (0.0-1.0). >=0.7=AUTO(green), >=0.4=REVIEW(yellow), <0.4=LOW(red).
 */
export const ConfidenceBadge = memo(function ConfidenceBadge({ confidence }: { confidence: number }) {
  const color = confidence >= 0.7 ? "var(--success)" : confidence >= 0.4 ? "var(--warning)" : "var(--error)";
  const label = confidence >= 0.7 ? "AUTO" : confidence >= 0.4 ? "REVIEW" : "LOW";
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 8px",
      borderRadius: 4,
      fontSize: 11,
      fontWeight: 600,
      color: "var(--bg)",
      background: color,
    }}>
      {label} ({confidence.toFixed(2)})
    </span>
  );
});
