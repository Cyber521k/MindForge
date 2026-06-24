import { useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";

interface LinePoint {
  label: string;
  value: number;
}

/**
 * SVG line chart for probe score over time.
 * Pure SVG with measured container width, no external deps.
 */
export function ScoreLineChart({ data, height = 160 }: { data: LinePoint[]; height?: number }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);

  useEffect(() => {
    if (!containerRef.current) return;
    const update = () => {
      if (containerRef.current) {
        setWidth(containerRef.current.clientWidth);
      }
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  if (data.length < 2) {
    return (
      <div style={{ padding: 20, color: "var(--text-dim)", fontSize: 13, textAlign: "center" }}>
        Need at least 2 probe runs to show score trend. Run more probes to see the timeline.
      </div>
    );
  }

  const padding = { top: 16, right: 16, bottom: 24, left: 36 };
  const h = height;
  const chartW = Math.max(width - padding.left - padding.right, 200);
  const chartH = h - padding.top - padding.bottom;

  const values = data.map((d) => d.value);
  const minVal = Math.min(...values, 0);
  const maxVal = Math.max(...values, 100);
  const valRange = maxVal - minVal || 1;

  const points = data.map((d, i) => {
    const x = padding.left + (i / (data.length - 1)) * chartW;
    const y = padding.top + chartH - ((d.value - minVal) / valRange) * chartH;
    return { x, y, ...d };
  });

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${padding.top + chartH} L ${points[0].x} ${padding.top + chartH} Z`;

  // Y-axis gridlines
  const gridLines = [0, 25, 50, 75, 100].filter((v) => v >= minVal && v <= maxVal);

  return (
    <div ref={containerRef} style={{ overflow: "visible" }}>
      <svg width="100%" height={h} style={{ overflow: "visible" }}>
        {/* Grid lines */}
        {gridLines.map((v) => {
          const y = padding.top + chartH - ((v - minVal) / valRange) * chartH;
          return (
            <g key={v}>
              <line x1={padding.left} y1={y} x2={padding.left + chartW} y2={y} stroke="var(--surface-raised)" strokeWidth={1} strokeDasharray="2 4" />
              <text x={padding.left - 6} y={y + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9}>
                {v}%
              </text>
            </g>
          );
        })}

        {/* Area fill */}
        <motion.path
          d={areaPath}
          fill="var(--accent)"
          fillOpacity={0.1}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        />

        {/* Line */}
        <motion.path
          d={linePath}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.6, ease: "easeInOut" }}
          style={{ filter: "drop-shadow(0 0 4px var(--accent-glow))" }}
        />

        {/* Points */}
        {points.map((p, i) => (
          <motion.circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={4}
            fill="var(--bg)"
            stroke="var(--accent)"
            strokeWidth={2}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.4 + i * 0.05 }}
          />
        ))}

        {/* X-axis labels */}
        {points.map((p, i) => (
          <text
            key={i}
            x={p.x}
            y={h - 6}
            textAnchor="middle"
            fill="var(--text-dim)"
            fontSize={9}
          >
            {p.label}
          </text>
        ))}
      </svg>
    </div>
  );
}
