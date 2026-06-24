export function ConfidenceBadge({ confidence }: { confidence: number }) {
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
}
