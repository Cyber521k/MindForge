import { motion } from "framer-motion";

interface BarDatum {
  label: string;
  correct: number;
  incorrect: number;
}

/**
 * Stacked horizontal bar chart: correct (green) vs incorrect (red) per subject.
 * Pure SVG, no external deps.
 */
export function SubjectBarChart({ data, height = 260 }: { data: BarDatum[]; height?: number }) {
  if (data.length === 0) {
    return (
      <div style={{ padding: 20, color: "var(--text-dim)", fontSize: 13, textAlign: "center" }}>
        No probe data yet. Run a probe to see results by subject.
      </div>
    );
  }

  const maxTotal = Math.max(...data.map((d) => d.correct + d.incorrect), 1);
  const barH = 22;
  const gap = 8;
  const labelW = 110;
  const chartW = 100; // percentage-based width units
  const totalH = data.length * (barH + gap) + 30;

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width="100%" height={Math.max(height, totalH)} style={{ minWidth: 400 }}>
        {data.map((d, i) => {
          const correctPct = (d.correct / maxTotal) * chartW;
          const incorrectPct = (d.incorrect / maxTotal) * chartW;
          const y = i * (barH + gap) + 4;
          return (
            <g key={d.label}>
              {/* Label */}
              <text
                x={labelW - 8}
                y={y + barH / 2 + 4}
                textAnchor="end"
                fill="var(--text-secondary)"
                fontSize={11}
                style={{ textTransform: "capitalize" }}
              >
                {d.label.length > 14 ? d.label.slice(0, 13) + "…" : d.label}
              </text>
              {/* Background track */}
              <rect
                x={labelW}
                y={y}
                width={`${chartW}%`}
                height={barH}
                fill="var(--surface-raised)"
                rx={3}
              />
              {/* Correct bar */}
              <motion.rect
                x={labelW}
                y={y}
                height={barH}
                rx={3}
                fill="var(--success)"
                initial={{ width: 0 }}
                animate={{ width: `${correctPct}%` }}
                transition={{ duration: 0.5, delay: i * 0.03 }}
              />
              {/* Incorrect bar (stacked) */}
              <motion.rect
                x={labelW + correctPct}
                y={y}
                height={barH}
                rx={3}
                fill="var(--error)"
                initial={{ width: 0 }}
                animate={{ width: `${incorrectPct}%` }}
                transition={{ duration: 0.5, delay: i * 0.03 + 0.15 }}
              />
              {/* Count labels */}
              <text
                x={labelW + correctPct + incorrectPct + 4}
                y={y + barH / 2 + 4}
                fill="var(--text-dim)"
                fontSize={10}
              >
                {d.correct + d.incorrect}
              </text>
            </g>
          );
        })}
      </svg>
      {/* Legend */}
      <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 11, color: "var(--text-secondary)" }}>
        <span>
          <span style={{ display: "inline-block", width: 10, height: 10, background: "var(--success)", borderRadius: 2, marginRight: 4 }} />
          Correct
        </span>
        <span>
          <span style={{ display: "inline-block", width: 10, height: 10, background: "var(--error)", borderRadius: 2, marginRight: 4 }} />
          Incorrect
        </span>
      </div>
    </div>
  );
}
