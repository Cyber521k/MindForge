import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiGet, apiPost, type TrainingEntry } from "../lib/api";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

export function ReviewDashboard({ onFormat }: { onFormat?: () => void }) {
  const [entries, setEntries] = useState<TrainingEntry[]>([]);
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [editedChosen, setEditedChosen] = useState("");
  const [editedRejected, setEditedRejected] = useState("");
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, accepted: 0, rejected: 0, edited: 0, skipped: 0 });
  const [showHelp, setShowHelp] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

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
      return; // Don't advance on error
    }
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
      // Toggle help overlay with ?
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        setShowHelp((s) => !s);
        return;
      }

      // Escape closes help overlay
      if (e.key === "Escape" && showHelp) {
        setShowHelp(false);
        return;
      }

      // Don't process other shortcuts if help is open
      if (showHelp) return;

      // Don't fire when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        // Allow Escape to exit edit mode
        if (e.key === "Escape") {
          setEditMode(false);
          (e.target as HTMLElement).blur();
        }
        return;
      }

      if (loading || !entries[idx]) return;

      // Ctrl+S to save (save edit if in edit mode, otherwise accept)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (editMode) {
          act("edit");
        } else {
          act("accept");
        }
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
        case "a":
          act("accept");
          break;
        case "r":
          act("reject");
          break;
        case "e":
          if (entries[idx]) {
            setEditedChosen(entries[idx].chosen);
            setEditedRejected(entries[idx].rejected);
            setEditMode(true);
          }
          break;
        case "s":
          act("skip");
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [idx, entries, loading, editMode, showHelp, navigate]);

  if (loading)
    return <LoadingState message="Loading review queue..." />;

  if (error)
    return (
      <div style={{ padding: 24 }}>
        <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Review Dashboard</h1>
        <ErrorState message={`Failed to load review queue: ${error}`} onRetry={load} />
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

      {/* Action error */}
      {actionError && (
        <div className="panel" style={{ padding: 12, marginBottom: 16, borderLeft: "3px solid var(--error)" }}>
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
            style={{ padding: 20, marginBottom: 20 }}
          >
            {/* Metadata */}
            <div style={{ display: "flex", gap: 12, marginBottom: 16, fontSize: 12, color: "var(--text-secondary)" }}>
              <span>[{entry.subject}]</span>
              <span>Format: {entry.format}</span>
              <ConfidenceBadge confidence={entry.confidence ?? 0.55} />
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
              <motion.button whileHover={{ scale: 1.05 }} onClick={() => setShowHelp(true)} className="btn-gold" style={{ background: "var(--surface-raised)", color: "var(--text-secondary)", padding: "10px 16px", borderRadius: 6, cursor: "pointer", fontWeight: 600, border: "1px solid var(--border)" }}>
                ? Help
              </motion.button>
            </div>

            {/* Keyboard hints */}
            <div style={{ marginTop: 12, fontSize: 11, color: "var(--text-dim)" }}>
              Shortcuts: ←→ Navigate · Enter=Accept · Del=Reject · E=Edit · S=Skip · Ctrl+S=Save · Press ? for shortcuts
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
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
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
              className="panel gold-glow"
              style={{ padding: 28, maxWidth: 480, width: "90%" }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <h2 style={{ fontSize: 18, color: "var(--accent)" }}>Keyboard Shortcuts</h2>
                <button
                  onClick={() => setShowHelp(false)}
                  style={{
                    background: "var(--surface-raised)",
                    color: "var(--text)",
                    border: "1px solid var(--border)",
                    borderRadius: 4,
                    cursor: "pointer",
                    padding: "2px 10px",
                    fontSize: 16,
                  }}
                >
                  ✕
                </button>
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
                      <td style={{ padding: "8px 0", color: "var(--accent)", fontWeight: 600, fontFamily: "monospace", whiteSpace: "nowrap" }}>
                        {key}
                      </td>
                      <td style={{ padding: "8px 0 8px 16px", color: "var(--text)" }}>
                        {action}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div style={{ marginTop: 16, fontSize: 11, color: "var(--text-dim)", textAlign: "center" }}>
                Press ? or Esc to close
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
