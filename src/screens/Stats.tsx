import { useState, useEffect, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import { apiGet, type Stats as StatsType, type ResponseEntry, type TrainingEntry } from "../lib/api";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";
import { SubjectBarChart } from "../components/SubjectBarChart";
import { ErrorTypePie } from "../components/ErrorTypePie";
import { ScoreLineChart } from "../components/ScoreLineChart";

// Colors for pie slices
const ERROR_COLORS: Record<string, string> = {
  factual_error: "var(--error)",
  close_confusion: "var(--warning)",
  reasoning_error: "var(--info)",
  other: "var(--text-dim)",
};
const ERROR_LABELS: Record<string, string> = {
  factual_error: "Factual Error",
  close_confusion: "Close Confusion",
  reasoning_error: "Reasoning Error",
  other: "Other",
};

export function Stats() {
  const [stats, setStats] = useState<StatsType | null>(null);
  const [responses, setResponses] = useState<ResponseEntry[]>([]);
  const [trainingEntries, setTrainingEntries] = useState<TrainingEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      apiGet<StatsType>("/api/stats"),
      apiGet<ResponseEntry[]>("/api/responses"),
      apiGet<TrainingEntry[]>("/api/training-entries"),
    ])
      .then(([s, r, t]) => {
        setStats(s);
        setResponses(r || []);
        setTrainingEntries(t || []);
        setLoading(false);
      })
      .catch((err) => {
        setError(err?.message || String(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Compute bar chart data: correct vs incorrect by subject
  const subjectData = useMemo(() => {
    const map = new Map<string, { correct: number; incorrect: number }>();
    for (const r of responses) {
      const subj = r.subject || "unknown";
      if (!map.has(subj)) map.set(subj, { correct: 0, incorrect: 0 });
      const entry = map.get(subj)!;
      if (r.is_correct) entry.correct++;
      else entry.incorrect++;
    }
    return Array.from(map.entries())
      .map(([label, v]) => ({ label: label.replace(/_/g, " "), ...v }))
      .sort((a, b) => b.correct + b.incorrect - (a.correct + a.incorrect))
      .slice(0, 12); // top 12 subjects
  }, [responses]);

  // Compute error type distribution pie chart
  const errorData = useMemo(() => {
    const counts: Record<string, number> = {
      factual_error: 0,
      close_confusion: 0,
      reasoning_error: 0,
      other: 0,
    };

    for (const r of responses) {
      if (r.is_correct) continue;
      // Classify errors based on confidence and answer proximity
      const modelAns = (r.model_answer_letter || "").toLowerCase();
      const correctAns = (r.correct_answer_letter || "").toLowerCase();
      if (modelAns && correctAns) {
        // Close confusion: model picked an adjacent answer (A↔B, C↔D)
        const modelIdx = modelAns.charCodeAt(0) - 97;
        const correctIdx = correctAns.charCodeAt(0) - 97;
        const dist = Math.abs(modelIdx - correctIdx);
        if (dist === 1) {
          counts.close_confusion++;
        } else if (r.confidence !== undefined && r.confidence < 0.3) {
          counts.factual_error++;
        } else {
          counts.reasoning_error++;
        }
      } else {
        counts.other++;
      }
    }

    return Object.entries(counts)
      .filter(([, v]) => v > 0)
      .map(([k, v]) => ({
        label: ERROR_LABELS[k] || k,
        value: v,
        color: ERROR_COLORS[k] || "var(--text-dim)",
      }));
  }, [responses]);

  // Compute score-over-time line chart from responses grouped by created_at
  const scoreTimeline = useMemo(() => {
    if (responses.length === 0) return [];

    // Group responses by model+date bucket (use created_at timestamp)
    const buckets = new Map<string, { correct: number; total: number; ts: number }>();

    for (const r of responses) {
      const ts = r.created_at || 0;
      if (!ts) continue;
      const model = r.model || "unknown";
      const date = new Date(ts * 1000);
      const key = `${model.slice(0, 15)}\n${date.toLocaleDateString()}`;
      if (!buckets.has(key)) buckets.set(key, { correct: 0, total: 0, ts });
      const b = buckets.get(key)!;
      b.total++;
      if (r.is_correct) b.correct++;
    }

    return Array.from(buckets.entries())
      .map(([label, v]) => ({
        label,
        value: v.total > 0 ? (v.correct / v.total) * 100 : 0,
      }))
      .sort((a, b) => {
        // Try to sort by timestamp embedded in label
        return 0; // keep insertion order (already by created_at DESC from API)
      })
      .reverse() // chronological order
      .slice(-15); // last 15 runs
  }, [responses]);

  // Memory tier breakdown (from stats accuracy domains)
  const tierBreakdown = useMemo(() => {
    const s = stats || ({} as StatsType);
    const domains = s.accuracy ? Object.keys(s.accuracy) : [];
    return domains.map((d) => ({
      domain: d,
      accuracy: s.accuracy[d] || 0,
    }));
  }, [stats]);

  // Summary card values
  const totalProbes = responses.length;
  const correctCount = responses.filter((r) => r.is_correct).length;
  const accuracyPct = totalProbes > 0 ? (correctCount / totalProbes) * 100 : 0;
  const corrections = trainingEntries.filter((e) => e.status === "edited").length;
  const trainingTotal = trainingEntries.length;

  if (loading)
    return <LoadingState message="Loading statistics..." />;

  if (error)
    return (
      <div style={{ padding: 24 }}>
        <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Statistics</h1>
        <ErrorState message={`Failed to load statistics: ${error}`} onRetry={load} />
      </div>
    );

  const s = stats || ({} as StatsType);

  return (
    <div style={{ padding: 24, height: "100%", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Statistics Dashboard</h1>

      {/* Summary Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 20 }}>
        <SummaryCard label="Total Probes" value={totalProbes} icon="🔍" color="var(--accent)" />
        <SummaryCard label="Accuracy" value={`${accuracyPct.toFixed(1)}%`} icon="✓" color={accuracyPct >= 70 ? "var(--success)" : accuracyPct >= 50 ? "var(--warning)" : "var(--error)"} />
        <SummaryCard label="Corrections" value={corrections} icon="✎" color="var(--info)" />
        <SummaryCard label="Training Entries" value={trainingTotal} icon="📦" color="var(--accent-secondary)" />
        <SummaryCard label="Subjects" value={`${s.subjects || 0}/57`} icon="📚" color="var(--accent)" />
        <SummaryCard label="Train Runs" value={s.training_runs || 0} icon="🎯" color="var(--accent-secondary)" />
      </div>

      {/* Bar Chart: Probe Results by Subject */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          📊 Probe Results by Subject (Correct vs Incorrect)
        </h3>
        <SubjectBarChart data={subjectData} />
      </div>

      {/* Two-column: Pie Chart + Memory Tier Breakdown */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
        {/* Pie Chart: Error Type Distribution */}
        <div className="panel" style={{ padding: 20 }}>
          <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
            🥧 Error Type Distribution
          </h3>
          <ErrorTypePie data={errorData} />
        </div>

        {/* Progress Bars: Accuracy by Domain (tier breakdown) */}
        <div className="panel" style={{ padding: 20 }}>
          <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
            📈 Accuracy by Domain
          </h3>
          {tierBreakdown.length === 0 ? (
            <div style={{ color: "var(--text-dim)", fontSize: 13, padding: 20, textAlign: "center" }}>
              No accuracy data yet. Run probes to see domain breakdown.
            </div>
          ) : (
            tierBreakdown.map((t, i) => (
              <motion.div
                key={t.domain}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                style={{ marginBottom: 12 }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                  <span style={{ textTransform: "capitalize" }}>{t.domain}</span>
                  <span style={{ color: t.accuracy >= 70 ? "var(--success)" : t.accuracy >= 50 ? "var(--warning)" : "var(--error)", fontWeight: 600 }}>
                    {t.accuracy.toFixed(1)}%
                  </span>
                </div>
                <div className="progress-bar" style={{ height: 10 }}>
                  <motion.div
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${t.accuracy}%` }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                    style={{ background: t.accuracy >= 70 ? "var(--success)" : t.accuracy >= 50 ? "var(--warning)" : "var(--error)" }}
                  />
                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>

      {/* Line Chart: Probe Score Over Time */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          📉 Probe Score Over Time
        </h3>
        <ScoreLineChart data={scoreTimeline} />
      </div>

      {/* Training History */}
      <div className="panel" style={{ padding: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          Training History
        </h3>
        {trainingEntries.length === 0 ? (
          <div style={{ color: "var(--text-dim)", fontSize: 13 }}>
            No training runs yet. Complete a fine-tuning run to see history here.
          </div>
        ) : (
          <div style={{ fontSize: 13 }}>
            <div style={{ display: "flex", gap: 20, marginBottom: 12, flexWrap: "wrap" }}>
              <span style={{ color: "var(--success)" }}>✓ Accepted: {trainingEntries.filter((e) => e.status === "accepted").length}</span>
              <span style={{ color: "var(--info)" }}>✎ Edited: {trainingEntries.filter((e) => e.status === "edited").length}</span>
              <span style={{ color: "var(--error)" }}>✗ Rejected: {trainingEntries.filter((e) => e.status === "rejected").length}</span>
              <span style={{ color: "var(--warning)" }}>◐ Pending: {trainingEntries.filter((e) => e.status === "pending").length}</span>
            </div>
            <div style={{ color: "var(--text-dim)", fontSize: 12 }}>
              {trainingTotal} total training entries · {s.training_runs || 0} fine-tuning run(s) completed
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/** Summary card component */
function SummaryCard({ label, value, icon, color }: { label: string; value: string | number; icon: string; color: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel"
      style={{ padding: 16, borderLeft: `3px solid ${color}` }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 18 }}>{icon}</span>
        <span style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>{label}</span>
      </div>
      <div style={{ fontSize: 24, fontWeight: 700, color }}>{value}</div>
    </motion.div>
  );
}
