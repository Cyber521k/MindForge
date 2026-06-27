import { useState, useEffect, useRef, useCallback, type CSSProperties } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiPost, apiGet } from "../lib/api";
import { useWebSocket } from "../hooks/useWebSocket";
import { ProgressRing } from "../components/ProgressRing";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

const XBOX = {
  primaryText: "#FFF8DC",
  neonGreen: "var(--xbox-neon-green, #00ff41)",
  chartreuse: "var(--xbox-chartreuse, #ccff00)",
  dimGreen: "var(--xbox-dim-green, #5f8f5f)",
  glow: "var(--xbox-glow, 0 0 18px rgba(0, 255, 65, 0.45))",
};

const screenStyle: CSSProperties = {
  padding: 24,
  paddingRight: 128,
  height: "100%",
  overflowY: "auto",
  position: "relative",
  color: XBOX.primaryText,
};

const titleStyle: CSSProperties = {
  fontSize: 24,
  marginBottom: 8,
  background: "linear-gradient(180deg, #C0C0C0, #808080)",
  backgroundClip: "text",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
  color: "transparent",
  fontFamily: "'Arial Black', Impact, sans-serif",
  fontWeight: 900,
  letterSpacing: 0,
  textTransform: "uppercase",
};

const headerGlowLineStyle: CSSProperties = {
  height: 1,
  marginBottom: 20,
  background: `linear-gradient(90deg, transparent, ${XBOX.neonGreen}, transparent)`,
  opacity: 0.6,
};

const xboxPanelStyle: CSSProperties = {
  clipPath: "polygon(12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%, 0 12px)",
  border: `1px solid ${XBOX.neonGreen}`,
  boxShadow: XBOX.glow,
  background: "rgba(10, 26, 10, 0.75)",
  backdropFilter: "blur(8px)",
  WebkitBackdropFilter: "blur(8px)",
  color: XBOX.primaryText,
};

const sectionLabelStyle: CSSProperties = {
  fontSize: 12,
  color: XBOX.neonGreen,
  marginBottom: 8,
  textTransform: "uppercase",
  letterSpacing: 0,
};

const decorativeIconStyle: CSSProperties = {
  position: "absolute",
  top: 24,
  right: 24,
  width: 80,
  height: 80,
  filter: `drop-shadow(${XBOX.glow})`,
  pointerEvents: "none",
};

function ProbingProgressIcon() {
  return (
    <div aria-hidden="true" style={decorativeIconStyle}>
      <div
        style={{
          position: "absolute",
          top: 6,
          left: 6,
          width: 48,
          height: 48,
          borderRadius: "50%",
          border: `5px solid ${XBOX.neonGreen}`,
          background: "radial-gradient(circle at 35% 30%, rgba(204,255,0,0.35), rgba(0,0,0,0.08) 55%, rgba(0,0,0,0.45))",
          boxShadow: `inset 0 0 18px rgba(0, 255, 65, 0.24), ${XBOX.glow}`,
        }}
      />
      <div
        style={{
          position: "absolute",
          width: 35,
          height: 8,
          right: 5,
          bottom: 15,
          background: `linear-gradient(90deg, ${XBOX.neonGreen}, ${XBOX.chartreuse})`,
          transform: "rotate(45deg)",
          transformOrigin: "left center",
          borderRadius: 8,
          boxShadow: XBOX.glow,
        }}
      />
    </div>
  );
}

function ScreenHeader() {
  return (
    <>
      <ProbingProgressIcon />
      <h1 style={titleStyle}>Probing Engine — Live</h1>
      <div style={headerGlowLineStyle} />
    </>
  );
}

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

  const stop = async () => {
    setRunning(false);
    setPaused(false);
    if (pollRef.current) clearInterval(pollRef.current);
    // Call cancel endpoint if we have a job ID
    if (jobId) {
      try {
        await apiPost(`/api/jobs/${jobId}/cancel`);
      } catch {
        // Best-effort cancel — server may not support it or job may already be done
      }
    }
  };

  return (
    <div style={screenStyle}>
      <ScreenHeader />

      {/* Error display */}
      {error && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 12, marginBottom: 16, borderLeft: "3px solid var(--error)" }}>
          <span style={{ color: "var(--error)", fontSize: 14, fontWeight: 600 }}>✗ {error}</span>
          <button
            onClick={() => { setError(null); start(); }}
            style={{ marginLeft: 12, padding: "4px 12px", fontSize: 12, background: "rgba(10, 26, 10, 0.7)", color: XBOX.primaryText, border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4, cursor: "pointer" }}
          >
            ↻ Retry
          </button>
        </div>
      )}

      {/* Config */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="Model"
            aria-label="Model name"
            style={{ flex: 1, padding: 8, background: "rgba(0, 0, 0, 0.24)", border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4, color: XBOX.primaryText }}
          />
          <input
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="Subject"
            aria-label="Subject"
            style={{ width: 200, padding: 8, background: "rgba(0, 0, 0, 0.24)", border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4, color: XBOX.primaryText }}
          />
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button
            className="btn-gold gold-glow"
            onClick={start}
            disabled={running}
            style={{ flex: 1, padding: 10, fontSize: 16, opacity: running ? 0.5 : 1, background: running ? "rgba(204, 255, 0, 0.15)" : undefined, borderLeft: running ? `3px solid ${XBOX.chartreuse}` : undefined, boxShadow: running ? XBOX.glow : undefined }}
          >
            {running ? "⏸ Probing..." : "► Start Probing"}
          </button>
          {running && (
            <>
              <button onClick={() => setPaused(!paused)} style={{ padding: "10px 20px", background: paused ? "rgba(204, 255, 0, 0.15)" : "var(--warning)", color: paused ? XBOX.chartreuse : "var(--bg)", border: paused ? `1px solid ${XBOX.chartreuse}` : "none", borderLeft: paused ? `3px solid ${XBOX.chartreuse}` : "none", boxShadow: paused ? XBOX.glow : "none", borderRadius: 6, cursor: "pointer", fontWeight: 600 }}>
                {paused ? "▶ Resume" : "⏸ Pause"}
              </button>
              <button onClick={stop} style={{ padding: "10px 20px", background: "var(--error)", color: XBOX.primaryText, border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 600 }}>
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
          <div className="panel" style={{ ...xboxPanelStyle, flex: 1, padding: 20 }}>
            <div style={{ fontSize: 14, marginBottom: 8 }}>
              Model: <span style={{ color: XBOX.chartreuse }}>{model.split("/").pop()}</span>
            </div>
            <div style={{ fontSize: 14, marginBottom: 8 }}>
              Subject: <span style={{ color: XBOX.chartreuse }}>{subject}</span>
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
        <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
          <div style={sectionLabelStyle}>
            Current Question
          </div>
          <div className="mono" style={{ fontSize: 14, padding: 12, background: "rgba(0, 0, 0, 0.24)", borderRadius: 4, marginBottom: 8, color: XBOX.primaryText }}>
            {currentQ || "Waiting for next question..."}
          </div>
          {currentA && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, color: XBOX.dimGreen, marginBottom: 4 }}>Model Response:</div>
              <div className="mono" style={{ padding: 12, background: "rgba(0, 0, 0, 0.24)", borderRadius: 4, fontSize: 13, color: XBOX.primaryText }}>
                {currentA}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recent Results */}
      {recentResults.length > 0 && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
          <div style={sectionLabelStyle}>
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
                <span style={{ color: XBOX.dimGreen, fontSize: 11 }}>[{r.subject}]</span>
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

      <div style={{ marginTop: 20, color: XBOX.dimGreen, fontSize: 12 }}>
        WebSocket: {connected ? "✓ Connected" : "✗ Disconnected (polling)"}
        {jobId && ` · Job: ${jobId}`}
      </div>
    </div>
  );
}
