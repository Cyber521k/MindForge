import { motion } from "framer-motion";

interface PieSlice {
  label: string;
  value: number;
  color: string;
}

/**
 * SVG donut/pie chart for error type distribution.
 * Pure SVG, no external deps.
 */
export function ErrorTypePie({ data, size = 180 }: { data: PieSlice[]; size?: number }) {
  const total = data.reduce((sum, d) => sum + d.value, 0);

  if (total === 0 || data.length === 0) {
    return (
      <div style={{ padding: 20, color: "var(--text-dim)", fontSize: 13, textAlign: "center" }}>
        No error data yet. Incorrect probe responses will appear here.
      </div>
    );
  }

  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 8;
  const innerR = r * 0.55; // donut hole

  let cumulativeAngle = -Math.PI / 2; // start at top

  const slices = data.map((d) => {
    const angle = (d.value / total) * Math.PI * 2;
    const startAngle = cumulativeAngle;
    const endAngle = cumulativeAngle + angle;
    cumulativeAngle = endAngle;

    // SVG arc path for donut slice
    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);
    const x3 = cx + innerR * Math.cos(endAngle);
    const y3 = cy + innerR * Math.sin(endAngle);
    const x4 = cx + innerR * Math.cos(startAngle);
    const y4 = cy + innerR * Math.sin(startAngle);
    const largeArc = angle > Math.PI ? 1 : 0;

    const path = `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} L ${x3} ${y3} A ${innerR} ${innerR} 0 ${largeArc} 0 ${x4} ${y4} Z`;

    return { ...d, path, pct: (d.value / total) * 100 };
  });

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap" }}>
      <svg width={size} height={size} style={{ flexShrink: 0 }}>
        {slices.map((s, i) => (
          <motion.path
            key={s.label}
            d={s.path}
            fill={s.color}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.08, duration: 0.3 }}
            style={{ transformOrigin: `${cx}px ${cy}px` }}
          />
        ))}
        {/* Center text */}
        <text x={cx} y={cy - 6} textAnchor="middle" fill="var(--accent)" fontSize={22} fontWeight={700}>
          {total}
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>
          errors
        </text>
      </svg>
      {/* Legend */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {slices.map((s) => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
            <span style={{ display: "inline-block", width: 10, height: 10, background: s.color, borderRadius: 2 }} />
            <span style={{ color: "var(--text)" }}>{s.label}</span>
            <span style={{ color: "var(--text-dim)" }}>{s.value} ({s.pct.toFixed(0)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}
