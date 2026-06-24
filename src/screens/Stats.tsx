import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { apiGet, type Stats as StatsType } from "../lib/api";

export function Stats() {
  const [stats, setStats] = useState<StatsType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGet<StatsType>("/api/stats")
      .then(setStats)
      .catch(() => setLoading(false));
  }, []);

  if (loading && !stats)
    return (
      <div style={{ padding: 40, color: "var(--text-secondary)" }}>
        <motion.div animate={{ opacity: [0.4, 1, 0.4] }} transition={{ duration: 1.5, repeat: Infinity }}>
          Loading statistics...
        </motion.div>
      </div>
    );

  const s = stats || ({} as StatsType);

  return (
    <div style={{ padding: 24, height: "100%", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Statistics</h1>

      {/* Overview */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>Overview</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12, fontSize: 14 }}>
          <div>Total Questions: <span style={{ color: "var(--accent)", fontWeight: 700 }}>{s.total_questions || 0}</span></div>
          <div>Training Pairs: <span style={{ color: "var(--accent)", fontWeight: 700 }}>{s.training_pairs || 0}</span></div>
          <div>Subjects Covered: <span style={{ color: "var(--accent)", fontWeight: 700 }}>{s.subjects || 0}/57</span></div>
          <div>Fine-Tuning Runs: <span style={{ color: "var(--accent)", fontWeight: 700 }}>{s.training_runs || 0}</span></div>
        </div>
      </div>

      {/* Accuracy by Domain */}
      {s.accuracy && Object.keys(s.accuracy).length > 0 && (
        <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
          <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
            Accuracy by Domain
          </h3>
          {Object.entries(s.accuracy).map(([domain, acc], i) => (
            <motion.div
              key={domain}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              style={{ marginBottom: 10 }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                <span style={{ textTransform: "capitalize" }}>{domain}</span>
                <span style={{ color: acc >= 70 ? "var(--success)" : acc >= 50 ? "var(--warning)" : "var(--error)", fontWeight: 600 }}>
                  {acc.toFixed(1)}%
                </span>
              </div>
              <div className="progress-bar" style={{ height: 8 }}>
                <motion.div
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${acc}%` }}
                  transition={{ duration: 0.5, delay: i * 0.05 }}
                  style={{ background: acc >= 70 ? "var(--success)" : acc >= 50 ? "var(--warning)" : "var(--error)" }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Training History placeholder */}
      <div className="panel" style={{ padding: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          Training History
        </h3>
        <div style={{ color: "var(--text-dim)", fontSize: 13 }}>
          No training runs yet. Complete a fine-tuning run to see history here.
        </div>
      </div>
    </div>
  );
}
