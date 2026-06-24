import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiGet, apiPost, type TrainingEntry } from "../lib/api";
import { ConfidenceBadge } from "../components/ConfidenceBadge";

export function ReviewDashboard({ onFormat }: { onFormat?: () => void }) {
  const [entries, setEntries] = useState<TrainingEntry[]>([]);
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [editedChosen, setEditedChosen] = useState("");
  const [editedRejected, setEditedRejected] = useState("");
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, accepted: 0, rejected: 0, edited: 0, skipped: 0 });

  const load = useCallback(() => {
    apiGet<TrainingEntry[]>("/api/training-entries")
      .then((e) => {
        setEntries(e);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const act = async (action: string) => {
    if (!entries[idx]) return;
    let body: any = { action };
    if (action === "edit") {
      body.edited_chosen = editedChosen;
      body.edited_rejected = editedRejected;
    }
    try {
      await apiPost(`/api/review/${entries[idx].id}`, body);
    } catch {}
    setSessionStats((s) => ({
      ...s,
      reviewed: s.reviewed + 1,
      [action === "accept" ? "accepted" : action === "reject" ? "rejected" : action === "edit" ? "edited" : "skipped"]:
        (s as any)[action === "accept" ? "accepted" : action === "reject" ? "rejected" : action === "edit" ? "edited" : "skipped"] + 1,
    }));
    setEditMode(false);
    setIdx(Math.min(idx + 1, entries.length));
    load();
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (loading || !entries[idx]) return;
      switch (e.key.toLowerCase()) {
        case "a": act("accept"); break;
        case "r": act("reject"); break;
        case "e":
          setEditedChosen(entries[idx].chosen);
          setEditedRejected(entries[idx].rejected);
          setEditMode(true);
          break;
        case "s": act("skip"); break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [idx, entries, loading]);

  if (loading)
    return (
      <div style={{ padding: 40, color: "var(--text-secondary)" }}>
        <motion.div animate={{ opacity: [0.4, 1, 0.4] }} transition={{ duration: 1.5, repeat: Infinity }}>
          Loading review queue...
        </motion.div>
      </div>
    );

  const entry = entries[idx];
  const remaining = entries.length - idx;

  return (
    <div style={{ padding: 24, height: "100%", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Review Dashboard</h1>

      {/* Queue Status */}
      <div className="panel" style={{ padding: 16, marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 14 }}>
            <span style={{ color: "var(--warning)" }}>{remaining}</span> items awaiting review
          </span>
          <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>
            Item {idx + 1} of {entries.length}
          </span>
        </div>
        <div className="progress-bar" style={{ height: 6 }}>
          <div className="progress-fill" style={{ width: `${entries.length > 0 ? (idx / entries.length) * 100 : 0}%` }} />
        </div>
      </div>

      <AnimatePresence mode="wait">
        {entry ? (
          <motion.div
            key={entry.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="panel"
            style={{ padding: 20, marginBottom: 20 }}
          >
            {/* Metadata */}
            <div style={{ display: "flex", gap: 12, marginBottom: 16, fontSize: 12, color: "var(--text-secondary)" }}>
              <span>[{entry.subject}]</span>
              <span>Format: {entry.format}</span>
              <ConfidenceBadge confidence={0.55} />
            </div>

            {/* Prompt */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Question</div>
              <div className="mono" style={{ padding: 12, background: "var(--surface-raised)", borderRadius: 4, fontSize: 13, whiteSpace: "pre-wrap" }}>
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
                  style={{ width: "100%", minHeight: 80, padding: 12, background: "var(--surface-raised)", border: "1px solid var(--success)", borderRadius: 4, color: "var(--text)", fontFamily: "monospace", fontSize: 13, resize: "vertical" }}
                />
              ) : (
                <div className="mono" style={{ padding: 12, background: "var(--surface-raised)", borderRadius: 4, fontSize: 13, borderLeft: "3px solid var(--success)" }}>
                  {entry.chosen}
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
                  style={{ width: "100%", minHeight: 80, padding: 12, background: "var(--surface-raised)", border: "1px solid var(--error)", borderRadius: 4, color: "var(--text)", fontFamily: "monospace", fontSize: 13, resize: "vertical" }}
                />
              ) : (
                <div className="mono" style={{ padding: 12, background: "var(--surface-raised)", borderRadius: 4, fontSize: 13, borderLeft: "3px solid var(--error)" }}>
                  {entry.rejected}
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
                <motion.button whileHover={{ scale: 1.05 }} onClick={() => act("edit")} className="btn-gold gold-glow" style={{ background: "var(--accent)", color: "var(--bg)", padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "none" }}>
                  ✓ Save Edit
                </motion.button>
              ) : (
                <motion.button whileHover={{ scale: 1.05 }} onClick={() => { setEditedChosen(entry.chosen); setEditedRejected(entry.rejected); setEditMode(true); }} className="btn-gold" style={{ background: "var(--info)", color: "var(--bg)", padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "none" }}>
                  ✎ Edit
                </motion.button>
              )}
              <motion.button whileHover={{ scale: 1.05 }} onClick={() => act("skip")} className="btn-gold" style={{ background: "var(--warning)", color: "var(--bg)", padding: "10px 20px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "none" }}>
                → Skip
              </motion.button>
            </div>

            {/* Keyboard hints */}
            <div style={{ marginTop: 12, fontSize: 11, color: "var(--text-dim)" }}>
              Shortcuts: A=Accept · R=Reject · E=Edit · S=Skip
            </div>
          </motion.div>
        ) : (
          <div className="panel" style={{ padding: 40, textAlign: "center" }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>⚕</div>
            <div style={{ color: "var(--text-secondary)", fontSize: 16, marginBottom: 8 }}>Review queue complete!</div>
            <div style={{ color: "var(--text-dim)", fontSize: 13, marginBottom: 20 }}>All entries have been reviewed.</div>
            <button className="btn-gold gold-glow" onClick={onFormat} style={{ padding: "12px 24px", fontSize: 16 }}>
              ► Continue to Format & Export
            </button>
          </div>
        )}
      </AnimatePresence>

      {/* Session Stats */}
      <div className="panel-raised" style={{ padding: 16, display: "flex", gap: 24, fontSize: 13 }}>
        <span>Reviewed: <span style={{ color: "var(--accent)" }}>{sessionStats.reviewed}</span></span>
        <span style={{ color: "var(--success)" }}>✓ {sessionStats.accepted}</span>
        <span style={{ color: "var(--error)" }}>✗ {sessionStats.rejected}</span>
        <span style={{ color: "var(--info)" }}>✎ {sessionStats.edited}</span>
        <span style={{ color: "var(--warning)" }}>→ {sessionStats.skipped}</span>
      </div>
    </div>
  );
}
