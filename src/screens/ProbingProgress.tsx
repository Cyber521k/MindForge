import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiPost, apiGet } from "../lib/api";
import { useWebSocket } from "../hooks/useWebSocket";
import { ProgressRing } from "../components/ProgressRing";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

interface ResultItem {
  subject: string;
  question: string;
  correct: boolean;
  confidence: number;
}

export function ProbingProgress({ onReview }: { onReview?: () => void }) {
  const [model, setModel] = useState("mlx-community/Llama-3.2-3B-Instruct-4bit");
  const [subject, setSubject] = useState("mathematics");
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState({ correct: 0, incorrect: 0, total: 0 });
  const [recentResults, setRecentResults] = useState<ResultItem[]>([]);
  const [currentQ, setCurrentQ] = useState<string>("");
  const [currentA, setCurrentA] = useState<string>("");
  const [jobId, setJobId] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const { connected, messages, latest } = useWebSocket();
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Watch latest WebSocket message
  useEffect(() => {
    if (!latest) return;
    if (latest.type === "progress" || latest.type === "probe_progress") {
      if (latest.progress !== undefined) setProgress(latest.progress);
      if (latest.correct !== undefined || latest.total !== undefined) {
        setResults({
          correct: latest.correct || 0,
          incorrect: latest.incorrect || 0,
          total: latest.total || 0,
        });
      }
      if (latest.question) setCurrentQ(latest.question);
      if (latest.response) setCurrentA(latest.response);
    }
    if (latest.type === "probe_result" || latest.type === "result") {
      setRecentResults((prev) =>
        [{ subject: latest.subject || subject, question: latest.question || "", correct: latest.correct || false, confidence: latest.confidence || 0 }, ...prev].slice(0, 10)
      );
    }
    if (latest.type === "job_complete" && latest.job_id === jobId) {
      setRunning(false);
      setProgress(100);
      if (pollRef.current) clearInterval(pollRef.current);
    }
    if (latest.type === "job_failed" && latest.job_id === jobId) {
      setRunning(false);
      setError(latest.error || "Probe job failed");
      if (pollRef.current) clearInterval(pollRef.current);
    }
  }, [latest, jobId, subject]);

  // Poll job status as fallback when WS isn't connected
  useEffect(() => {
    if (!running || !jobId || connected) return;
    pollRef.current = setInterval(async () => {
      try {
        const status = await apiGet(`/api/probe/${jobId}`);
        if (status.status === "completed") {
          setRunning(false);
          setProgress(100);
          if (pollRef.current) clearInterval(pollRef.current);
        } else if (status.status === "failed") {
          setRunning(false);
          setError(status.error || "Probe job failed");
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (err: any) {
        // Don't set error on poll failure — might be transient
      }
    }, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [running, jobId, connected]);

  const start = async () => {
    setRunning(true);
    setPaused(false);
    setProgress(0);
    setResults({ correct: 0, incorrect: 0, total: 0 });
    setRecentResults([]);
    setError(null);
    try {
      const res = await apiPost("/api/probe", { model, subject, tier: "1", limit: 25 });
      setJobId(res.job_id);
    } catch (err: any) {
      setRunning(false);
      setError(err?.message || String(err));
    }
  };

  const stop = () => {
    setRunning(false);
    setPaused(false);
    if (pollRef.current) clearInterval(pollRef.current);
  };

  return (
    <div style={{ padding: 24, height: "100%", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Probing Engine — Live</h1>

      {/* Error display */}
      {error && (
        <div className="panel" style={{ padding: 12, marginBottom: 16, borderLeft: "3px solid var(--error)" }}>
          <span style={{ color: "var(--error)", fontSize: 14, fontWeight: 600 }}>✗ {error}</span>
          <button
            onClick={() => { setError(null); start(); }}
            style={{ marginLeft: 12, padding: "4px 12px", fontSize: 12, background: "var(--surface-raised)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer" }}
          >
            ↻ Retry
          </button>
        </div>
      )}

      {/* Config */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="Model"
            style={{ flex: 1, padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }}
          />
          <input
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="Subject"
            style={{ width: 200, padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }}
          />
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button className="btn-gold gold-glow" onClick={start} disabled={running} style={{ flex: 1, padding: 10, fontSize: 16, opacity: running ? 0.5 : 1 }}>
            {running ? "⏸ Probing..." : "► Start Probing"}
          </button>
          {running && (
            <>
              <button onClick={() => setPaused(!paused)} style={{ padding: "10px 20px", background: "var(--warning)", color: "var(--bg)", border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 600 }}>
                {paused ? "▶ Resume" : "⏸ Pause"}
              </button>
              <button onClick={stop} style={{ padding: "10px 20px", background: "var(--error)", color: "var(--text)", border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 600 }}>
                ⏹ Stop
              </button>
            </>
          )}
        </div>
      </div>

      {/* Progress + Stats */}
      {(running || progress > 0) && (
        <div style={{ display: "flex", gap: 20, alignItems: "flex-start", marginBottom: 20 }}>
          <ProgressRing value={progress} label="Progress" />
          <div className="panel" style={{ flex: 1, padding: 20 }}>
            <div style={{ fontSize: 14, marginBottom: 8 }}>
              Model: <span style={{ color: "var(--accent)" }}>{model.split("/").pop()}</span>
            </div>
            <div style={{ fontSize: 14, marginBottom: 8 }}>
              Subject: <span style={{ color: "var(--accent)" }}>{subject}</span>
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 12 }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--success)" }}>
                ✓ {results.correct}
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--error)" }}>
                ✗ {results.incorrect}
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--warning)" }}>
                ◐ {results.total - results.correct - results.incorrect}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Current Question */}
      {(running || currentQ) && (
        <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
            Current Question
          </div>
          <div className="mono" style={{ fontSize: 14, padding: 12, background: "var(--surface-raised)", borderRadius: 4, marginBottom: 8 }}>
            {currentQ || "Waiting for next question..."}
          </div>
          {currentA && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>Model Response:</div>
              <div className="mono" style={{ padding: 12, background: "var(--surface-raised)", borderRadius: 4, fontSize: 13, color: "var(--text)" }}>
                {currentA}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recent Results */}
      {recentResults.length > 0 && (
        <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
            Recent Results
          </div>
          <AnimatePresence>
            {recentResults.map((r, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 0", fontSize: 13, borderBottom: "1px solid var(--surface-raised)" }}
              >
                <span style={{ color: r.correct ? "var(--success)" : "var(--error)", fontWeight: 700 }}>
                  {r.correct ? "✓" : "✗"}
                </span>
                <span style={{ color: "var(--text-secondary)", fontSize: 11 }}>[{r.subject}]</span>
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.question}</span>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Stop & Review */}
      {(running || progress > 0) && (
        <button
          className="btn-gold gold-glow"
          onClick={onReview}
          style={{ width: "100%", padding: 12, fontSize: 16 }}
        >
          ⏹ Stop & Review Now
        </button>
      )}

      <div style={{ marginTop: 20, color: "var(--text-dim)", fontSize: 12 }}>
        WebSocket: {connected ? "✓ Connected" : "✗ Disconnected (polling)"}
        {jobId && ` · Job: ${jobId}`}
      </div>
    </div>
  );
}
