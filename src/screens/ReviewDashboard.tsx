import { useState, useEffect, useCallback, type CSSProperties } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiGet, apiPost, type TrainingEntry } from "../lib/api";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";
import { EmptyState } from "../components/EmptyState";
import { useWebSocket } from "../hooks/useWebSocket";

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
  marginBottom: 4,
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

function ReviewDashboardIcon() {
  return (
    <div aria-hidden="true" style={decorativeIconStyle}>
      <div
        style={{
          position: "absolute",
          inset: "12px 10px 4px",
          background: "linear-gradient(145deg, rgba(18, 48, 18, 0.95), rgba(4, 14, 4, 0.95))",
          border: `2px solid ${XBOX.neonGreen}`,
          clipPath: "polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px)",
          boxShadow: `inset 0 0 18px rgba(0, 255, 65, 0.18), ${XBOX.glow}`,
        }}
      />
      <div style={{ position: "absolute", top: 4, left: 24, width: 32, height: 16, borderRadius: 4, background: XBOX.chartreuse, boxShadow: XBOX.glow }} />
      {[26, 38, 50].map((top) => (
        <div key={top} style={{ position: "absolute", top, left: 22, right: 20, height: 3, background: XBOX.neonGreen, opacity: 0.75 }} />
      ))}
    </div>
  );
}

function ScreenHeader() {
  return (
    <>
      <ReviewDashboardIcon />
      <h1 style={titleStyle}>Review Dashboard</h1>
      <div style={headerGlowLineStyle} />
    </>
  );
}

// Judge model options
const JUDGE_MODELS = [
  { value: "auto", label: "Auto-detect (best available)" },
  { value: "gpt-4o", label: "GPT-4o (OpenAI)" },
  { value: "gpt-4o-mini", label: "GPT-4o mini (OpenAI)" },
  { value: "openrouter/anthropic/claude-3.5-sonnet", label: "Claude 3.5 Sonnet (OpenRouter)" },
  { value: "openrouter/meta-llama/llama-3.3-70b-instruct", label: "Llama 3.3 70B (OpenRouter)" },
  { value: "ollama/llama3.2", label: "Llama 3.2 (Ollama local)" },
];

interface AutoReviewResult {
  action: string;
  confidence: number;
  explanation: string;
  edited_chosen?: string | null;
  edited_rejected?: string | null;
  web_source?: {
    source_url: string;
    snippet: string;
    answer: string;
  } | null;
}

export function ReviewDashboard({ onFormat }: { onFormat?: () => void }) {
  const [entries, setEntries] = useState<TrainingEntry[]>([]);
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [editedChosen, setEditedChosen] = useState("");
  const [editedRejected, setEditedRejected] = useState("");
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, accepted: 0, rejected: 0, edited: 0, skipped: 0, autoReviewed: 0 });
  const [showHelp, setShowHelp] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // Auto-review state
  const [autoReviewing, setAutoReviewing] = useState(false);
  const [autoProgress, setAutoProgress] = useState(0);
  const [autoError, setAutoError] = useState<string | null>(null);
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const [judgeModel, setJudgeModel] = useState("auto");
  const [autoResults, setAutoResults] = useState<Record<number, AutoReviewResult>>({});

  const { latest } = useWebSocket();

  // Watch WebSocket for auto-review progress
  useEffect(() => {
    if (!latest || !autoReviewing) return;
    if (latest.type === "auto_review_progress") {
      setAutoProgress(latest.progress || 0);
    }
    if (latest.type === "auto_review_complete") {
      setAutoReviewing(false);
      setAutoProgress(100);
      // Merge results into entries
      if (latest.results) {
        const resultMap: Record<number, AutoReviewResult> = {};
        for (const r of latest.results) {
          if (r.entry_id) resultMap[r.entry_id] = r;
        }
        setAutoResults(resultMap);
      }
      // Reload entries to get updated statuses
      load();
    }
    if (latest.type === "auto_review_failed") {
      setAutoReviewing(false);
      setAutoError(latest.error || "Auto-review failed");
    }
  }, [latest, autoReviewing]);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    apiGet<TrainingEntry[]>("/api/training-entries")
      .then((e) => {
        setEntries(e);
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

  const act = async (action: string) => {
    if (!entries[idx]) return;
    setActionError(null);
    let body: any = { action };
    if (action === "edit") {
      body.edited_chosen = editedChosen;
      body.edited_rejected = editedRejected;
    }
    try {
      await apiPost(`/api/review/${entries[idx].id}`, body);
    } catch (err: any) {
      setActionError(err?.message || String(err));
      return;
    }
    const newStatus = action === "accept" ? "accepted"
      : action === "reject" ? "rejected"
      : action === "edit" ? "edited"
      : "pending";
    setEntries((prev) => prev.map((e, i) =>
      i === idx ? { ...e, status: newStatus, ...(action === "edit" ? { chosen: editedChosen, rejected: editedRejected } : {}) } : e
    ));
    setSessionStats((s) => ({
      ...s,
      reviewed: s.reviewed + 1,
      [action === "accept" ? "accepted" : action === "reject" ? "rejected" : action === "edit" ? "edited" : "skipped"]:
        (s as any)[action === "accept" ? "accepted" : action === "reject" ? "rejected" : action === "edit" ? "edited" : "skipped"] + 1,
    }));
    setEditMode(false);
    setIdx(Math.min(idx + 1, entries.length));
  };

  const startAutoReview = async () => {
    setAutoReviewing(true);
    setAutoProgress(0);
    setAutoError(null);
    setAutoResults({});
    try {
      const res = await apiPost("/api/review/auto", {
        web_search: webSearchEnabled,
        judge_model: judgeModel,
      });
      // If the endpoint returns results synchronously (not via WS)
      if (res.results) {
        const resultMap: Record<number, AutoReviewResult> = {};
        for (const r of res.results) {
          if (r.entry_id) resultMap[r.entry_id] = r;
        }
        setAutoResults(resultMap);
        setAutoReviewing(false);
        setAutoProgress(100);
        load();
      }
      // If it returns a job_id, progress comes via WebSocket
    } catch (err: any) {
      setAutoReviewing(false);
      setAutoError(err?.message || String(err));
    }
  };

  // Navigate prev/next without acting
  const navigate = useCallback(
    (dir: "prev" | "next") => {
      setEditMode(false);
      setActionError(null);
      if (dir === "prev" && idx > 0) setIdx(idx - 1);
      if (dir === "next" && idx < entries.length - 1) setIdx(idx + 1);
    },
    [idx, entries.length]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        setShowHelp((s) => !s);
        return;
      }
      if (e.key === "Escape" && showHelp) {
        setShowHelp(false);
        return;
      }
      if (showHelp) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) {
        if (e.key === "Escape") {
          setEditMode(false);
          (e.target as HTMLElement).blur();
        }
        return;
      }
      if (loading || !entries[idx]) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (editMode) act("edit");
        else act("accept");
        return;
      }
      switch (e.key) {
        case "ArrowLeft":
        case "ArrowUp":
          e.preventDefault();
          navigate("prev");
          break;
        case "ArrowRight":
        case "ArrowDown":
          e.preventDefault();
          navigate("next");
          break;
        case "Enter":
          e.preventDefault();
          act("accept");
          break;
        case "Delete":
        case "Backspace":
          e.preventDefault();
          act("reject");
          break;
      }
      switch (e.key.toLowerCase()) {
        case "a": act("accept"); break;
        case "r": act("reject"); break;
        case "e":
          if (entries[idx]) {
            setEditedChosen(entries[idx].chosen);
            setEditedRejected(entries[idx].rejected);
            setEditMode(true);
          }
          break;
        case "s": act("skip"); break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [idx, entries, loading, editMode, showHelp, navigate]);

  if (loading)
    return <LoadingState message="Loading review queue..." />;

  if (error)
    return (
      <div style={screenStyle}>
        <ScreenHeader />
        <ErrorState message={`Failed to load review queue: ${error}`} onRetry={load} />
      </div>
    );

  const entry = entries[idx];
  const remaining = entries.length - idx;
  const autoResult = entry ? autoResults[entry.id] : undefined;

  if (entries.length === 0) {
    return (
      <div style={screenStyle}>
        <ScreenHeader />
        <div className="panel" style={{ ...xboxPanelStyle, padding: 24 }}>
          <EmptyState
            icon="📋"
            title="Review queue is empty"
            message="Run a probe to generate training entries that need review."
          />
        </div>
      </div>
    );
  }

  return (
    <div style={screenStyle}>
      <ScreenHeader />

      {/* Auto-Review Controls */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 16, marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            {/* Judge model selector */}
            <div>
              <label style={{ fontSize: 11, color: XBOX.dimGreen, display: "block", marginBottom: 2 }}>Judge Model</label>
              <select
                value={judgeModel}
                onChange={(e) => setJudgeModel(e.target.value)}
                disabled={autoReviewing}
                style={{
                  padding: "6px 8px",
                  background: "rgba(0, 0, 0, 0.24)",
                  border: `1px solid ${XBOX.neonGreen}`,
                  borderRadius: 4,
                  color: XBOX.primaryText,
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                {JUDGE_MODELS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>

            {/* Web search toggle */}
            <div>
              <label style={{ fontSize: 11, color: XBOX.dimGreen, display: "block", marginBottom: 2 }}>Web Search</label>
              <button
                onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                disabled={autoReviewing}
                aria-pressed={webSearchEnabled}
                aria-label={`Web search ${webSearchEnabled ? "enabled" : "disabled"}`}
                style={{
                  padding: "6px 12px",
                  background: webSearchEnabled ? "rgba(204, 255, 0, 0.15)" : "rgba(10, 26, 10, 0.7)",
                  color: webSearchEnabled ? XBOX.chartreuse : XBOX.dimGreen,
                  border: webSearchEnabled ? `1px solid ${XBOX.chartreuse}` : `1px solid ${XBOX.neonGreen}`,
                  borderLeft: webSearchEnabled ? `3px solid ${XBOX.chartreuse}` : `1px solid ${XBOX.neonGreen}`,
                  boxShadow: webSearchEnabled ? XBOX.glow : "none",
                  borderRadius: 4,
                  cursor: "pointer",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                {webSearchEnabled ? "🌐 ON" : "🌐 OFF"}
              </button>
            </div>
          </div>

          {/* Auto Review button */}
          <button
            className="btn-gold gold-glow"
            onClick={startAutoReview}
            disabled={autoReviewing}
            style={{ padding: "8px 20px", fontSize: 14, opacity: autoReviewing ? 0.5 : 1 }}
          >
            {autoReviewing ? "⏳ Reviewing..." : "🤖 Auto Review All"}
          </button>
        </div>

        {/* Auto-review progress bar */}
        {autoReviewing && (
          <div style={{ marginTop: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: XBOX.dimGreen, marginBottom: 4 }}>
              <span>Auto-reviewing entries...</span>
              <span>{autoProgress.toFixed(0)}%</span>
            </div>
            <div
              className="progress-bar"
              role="progressbar"
              aria-valuenow={Math.round(autoProgress)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Auto-review progress"
              style={{ height: 8 }}
            >
              <motion.div
                className="progress-fill"
                animate={{ width: `${autoProgress}%` }}
                style={{ width: `${autoProgress}%`, background: "var(--info)" }}
              />
            </div>
          </div>
        )}

        {/* Auto-review error */}
        {autoError && (
          <div style={{ marginTop: 12, padding: 10, borderLeft: "3px solid var(--error)", color: "var(--error)", fontSize: 13, fontWeight: 600 }}>
            ✗ Auto-review failed: {autoError}
          </div>
        )}
      </div>

      {/* Queue Status */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 16, marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 14 }}>
            <span style={{ color: "var(--warning)" }}>{remaining}</span> items awaiting review
          </span>
          <span style={{ fontSize: 14, color: XBOX.dimGreen }}>
            Item {idx + 1} of {entries.length}
          </span>
        </div>
        <div
          className="progress-bar"
          role="progressbar"
          aria-valuenow={entries.length > 0 ? Math.round((idx / entries.length) * 100) : 0}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Review progress: ${idx + 1} of ${entries.length}`}
          style={{ height: 6 }}
        >
          <div className="progress-fill" style={{ width: `${entries.length > 0 ? (idx / entries.length) * 100 : 0}%` }} />
        </div>
      </div>

      {/* Action error */}
      {actionError && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 12, marginBottom: 16, borderLeft: "3px solid var(--error)" }}>
          <span style={{ color: "var(--error)", fontSize: 14, fontWeight: 600 }}>✗ Action failed: {actionError}</span>
        </div>
      )}

      <AnimatePresence mode="wait">
        {entry ? (
          <motion.div
            key={entry.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="panel"
            style={{ ...xboxPanelStyle, padding: 20, marginBottom: 16 }}
          >
            {/* Metadata */}
            <div style={{ display: "flex", gap: 12, marginBottom: 16, fontSize: 12, color: XBOX.dimGreen, flexWrap: "wrap", alignItems: "center" }}>
              <span>[{entry.subject}]</span>
              <span>Format: {entry.format}</span>
              <ConfidenceBadge confidence={entry.confidence ?? 0.55} />
              {autoResult && (
                <span style={{
                  padding: "2px 8px",
                  borderRadius: 4,
                  fontSize: 10,
                  fontWeight: 600,
                  background: "var(--info)",
                  color: "var(--bg)",
                }}>
                  🤖 AUTO-REVIEWED
                </span>
              )}
            </div>

            {/* Auto-review result panel */}
            {autoResult && (
              <div className="panel-raised" style={{ ...xboxPanelStyle, padding: 14, marginBottom: 16, borderLeft: "3px solid var(--info)" }}>
                {/* Judge confidence meter */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                  <span style={{ fontSize: 12, color: XBOX.neonGreen, textTransform: "uppercase", letterSpacing: 0 }}>Judge Confidence</span>
                  <ConfidenceMeter confidence={autoResult.confidence} />
                </div>

                {/* Explanation */}
                {autoResult.explanation && (
                  <div style={{ fontSize: 13, color: XBOX.primaryText, marginBottom: 10, lineHeight: 1.5 }}>
                    <span style={{ color: XBOX.dimGreen, fontWeight: 600 }}>Explanation: </span>
                    {autoResult.explanation}
                  </div>
                )}

                {/* Web source indicator */}
                {autoResult.web_source && autoResult.web_source.source_url && (
                  <div style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "8px 12px",
                    background: "rgba(0, 0, 0, 0.24)",
                    borderRadius: 4,
                    fontSize: 12,
                  }}>
                    <span style={{ fontSize: 16 }}>🌐</span>
                    <div>
                      <div style={{ color: "var(--info)", fontWeight: 600 }}>Web-sourced correction</div>
                      <a
                        href={autoResult.web_source.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: "var(--accent-dim)", textDecoration: "none", fontSize: 11, wordBreak: "break-all" }}
                      >
                        {autoResult.web_source.source_url}
                      </a>
                      {autoResult.web_source.snippet && (
                        <div style={{ color: "var(--text-dim)", fontSize: 11, marginTop: 4, fontStyle: "italic" }}>
                          "{autoResult.web_source.snippet.slice(0, 200)}{autoResult.web_source.snippet.length > 200 ? "..." : ""}"
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Auto action badge */}
                <div style={{ marginTop: 8, fontSize: 12 }}>
                  <span style={{ color: XBOX.dimGreen }}>Auto action: </span>
                  <span style={{
                    fontWeight: 600,
                    color: autoResult.action === "accept" ? "var(--success)" : autoResult.action === "reject" ? "var(--error)" : "var(--info)",
                  }}>
                    {autoResult.action.toUpperCase()}
                  </span>
                </div>
              </div>
            )}

            {/* Prompt */}
            <div style={{ marginBottom: 16 }}>
              <div style={sectionLabelStyle}>Question</div>
              <div className="mono" style={{ padding: 12, background: "rgba(0, 0, 0, 0.24)", borderRadius: 4, fontSize: 13, whiteSpace: "pre-wrap", color: XBOX.primaryText }}>
                {entry.prompt}
              </div>
            </div>

            {/* Chosen (correct) */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: "var(--success)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>✓ Chosen (correct)</div>
              {editMode ? (
                <textarea
                  value={editedChosen}
                  onChange={(e) => setEditedChosen(e.target.value)}
                  style={{ width: "100%", minHeight: 80, padding: 12, background: "rgba(0, 0, 0, 0.24)", border: "1px solid var(--success)", borderRadius: 4, color: XBOX.primaryText, fontFamily: "monospace", fontSize: 13, resize: "vertical" }}
                />
              ) : (
                <div className="mono" style={{ padding: 12, background: "rgba(0, 0, 0, 0.24)", borderRadius: 4, fontSize: 13, borderLeft: "3px solid var(--success)", color: XBOX.primaryText }}>
                  {autoResult?.edited_chosen || entry.chosen}
                </div>
              )}
            </div>

            {/* Rejected (wrong) */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: "var(--error)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>✗ Rejected (wrong)</div>
              {editMode ? (
                <textarea
                  value={editedRejected}
                  onChange={(e) => setEditedRejected(e.target.value)}
                  style={{ width: "100%", minHeight: 80, padding: 12, background: "rgba(0, 0, 0, 0.24)", border: "1px solid var(--error)", borderRadius: 4, color: XBOX.primaryText, fontFamily: "monospace", fontSize: 13, resize: "vertical" }}
                />
              ) : (
                <div className="mono" style={{ padding: 12, background: "rgba(0, 0, 0, 0.24)", borderRadius: 4, fontSize: 13, borderLeft: "3px solid var(--error)", color: XBOX.primaryText }}>
                  {autoResult?.edited_rejected || entry.rejected}
                </div>
              )}
            </div>

            {/* Actions */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <motion.button whileHover={{ scale: 1.05 }} onClick={() => act("accept")} className="btn-gold" style={{ background: "var(--success)", color: "var(--bg)", padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "none" }}>
                ✓ Accept
              </motion.button>
              <motion.button whileHover={{ scale: 1.05 }} onClick={() => act("reject")} className="btn-gold" style={{ background: "var(--error)", color: "var(--text)", padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "none" }}>
                ✗ Reject
              </motion.button>
              {editMode ? (
                <motion.button whileHover={{ scale: 1.05 }} onClick={() => act("edit")} className="btn-gold gold-glow" style={{ background: "rgba(204, 255, 0, 0.15)", color: XBOX.chartreuse, padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: `1px solid ${XBOX.chartreuse}`, borderLeft: `3px solid ${XBOX.chartreuse}`, boxShadow: XBOX.glow }}>
                  ✓ Save Edit
                </motion.button>
              ) : (
                <motion.button whileHover={{ scale: 1.05 }} onClick={() => { setEditedChosen(autoResult?.edited_chosen || entry.chosen); setEditedRejected(autoResult?.edited_rejected || entry.rejected); setEditMode(true); }} className="btn-gold" style={{ background: "var(--info)", color: "var(--bg)", padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "none" }}>
                  ✎ Edit
                </motion.button>
              )}
              <motion.button whileHover={{ scale: 1.05 }} onClick={() => act("skip")} className="btn-gold" style={{ background: "var(--warning)", color: "var(--bg)", padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "none" }}>
                → Skip
              </motion.button>
              <motion.button whileHover={{ scale: 1.05 }} onClick={() => setShowHelp(true)} className="btn-gold" style={{ background: "rgba(10, 26, 10, 0.7)", color: XBOX.dimGreen, padding: "10px 16px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: `1px solid ${XBOX.neonGreen}` }}>
                ? Help
              </motion.button>
            </div>

            {/* Keyboard hints */}
            <div style={{ marginTop: 12, fontSize: 11, color: XBOX.dimGreen }}>
              Shortcuts: ←→ Navigate · Enter=Accept · Del=Reject · E=Edit · S=Skip · Ctrl+S=Save · Press ? for shortcuts
            </div>
          </motion.div>
        ) : (
          <div className="panel" style={{ ...xboxPanelStyle, padding: 40, textAlign: "center" }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>⚕</div>
            <div style={{ color: XBOX.neonGreen, fontSize: 16, marginBottom: 8 }}>Review queue complete!</div>
            <div style={{ color: XBOX.dimGreen, fontSize: 13, marginBottom: 20 }}>All entries have been reviewed.</div>
            <button className="btn-gold gold-glow" onClick={onFormat} style={{ padding: "12px 24px", fontSize: 16 }}>
              ► Continue to Format & Export
            </button>
          </div>
        )}
      </AnimatePresence>

      {/* Session Stats */}
      <div className="panel-raised" style={{ ...xboxPanelStyle, padding: 16, display: "flex", gap: 20, fontSize: 13, flexWrap: "wrap" }}>
        <span>Reviewed: <span style={{ color: XBOX.chartreuse }}>{sessionStats.reviewed}</span></span>
        <span style={{ color: "var(--success)" }}>✓ {sessionStats.accepted}</span>
        <span style={{ color: "var(--error)" }}>✗ {sessionStats.rejected}</span>
        <span style={{ color: "var(--info)" }}>✎ {sessionStats.edited}</span>
        <span style={{ color: "var(--warning)" }}>→ {sessionStats.skipped}</span>
        <span style={{ color: "var(--info)" }}>🤖 {sessionStats.autoReviewed}</span>
      </div>

      {/* Keyboard Shortcuts Help Overlay */}
      <AnimatePresence>
        {showHelp && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowHelp(false)}
            style={{
              position: "fixed",
              top: 0, left: 0, right: 0, bottom: 0,
              background: "rgba(0,0,0,0.7)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
            }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby="help-title"
              className="panel gold-glow"
              style={{ ...xboxPanelStyle, padding: 28, maxWidth: 480, width: "90%" }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <h2 id="help-title" style={{ fontSize: 18, color: XBOX.neonGreen }}>Keyboard Shortcuts</h2>
                <button
                  onClick={() => setShowHelp(false)}
                  aria-label="Close help dialog"
                  style={{
                    background: "rgba(10, 26, 10, 0.7)", color: XBOX.primaryText,
                    border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4,
                    cursor: "pointer", padding: "2px 10px", fontSize: 16,
                  }}
                >✕</button>
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <tbody>
                  {[
                    { key: "← / ↑", action: "Previous item" },
                    { key: "→ / ↓", action: "Next item" },
                    { key: "Enter", action: "Accept (approve)" },
                    { key: "Delete / Backspace", action: "Reject" },
                    { key: "Ctrl+S", action: "Save review (save edit or accept)" },
                    { key: "A", action: "Accept" },
                    { key: "R", action: "Reject" },
                    { key: "E", action: "Edit" },
                    { key: "S", action: "Skip" },
                    { key: "?", action: "Toggle this help" },
                    { key: "Esc", action: "Close help / exit edit" },
                  ].map(({ key, action }) => (
                    <tr key={key} style={{ borderBottom: "1px solid var(--surface-raised)" }}>
                      <td style={{ padding: "8px 0", color: XBOX.chartreuse, fontWeight: 600, fontFamily: "monospace", whiteSpace: "nowrap" }}>{key}</td>
                      <td style={{ padding: "8px 0 8px 16px", color: XBOX.primaryText }}>{action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ marginTop: 16, fontSize: 11, color: XBOX.dimGreen, textAlign: "center" }}>
                Press ? or Esc to close
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Color-coded confidence meter — green >=0.7, yellow 0.4-0.7, red <0.4 */
function ConfidenceMeter({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color = confidence >= 0.7 ? "var(--success)" : confidence >= 0.4 ? "var(--warning)" : "var(--error)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, maxWidth: 300 }}>
      <div className="progress-bar" style={{ flex: 1, height: 10 }}>
        <motion.div
          className="progress-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.4 }}
          style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}` }}
        />
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color, minWidth: 40, textAlign: "right" }}>
        {pct}%
      </span>
    </div>
  );
}
