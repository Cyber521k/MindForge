import { useState, useEffect, useMemo, useCallback, type CSSProperties } from "react";
import { motion } from "framer-motion";
import { apiGet, apiPost, type TrainingEntry } from "../lib/api";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";
import { EmptyState } from "../components/EmptyState";

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

const sectionHeadingStyle: CSSProperties = {
  marginBottom: 12,
  fontSize: 14,
  color: XBOX.neonGreen,
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

function FormatExportIcon() {
  return (
    <div aria-hidden="true" style={decorativeIconStyle}>
      <div
        style={{
          position: "absolute",
          inset: 4,
          borderRadius: "50%",
          background: `conic-gradient(from 20deg, #f4ffd9, ${XBOX.chartreuse}, #083b12, ${XBOX.neonGreen}, #f4ffd9)`,
          border: `2px solid ${XBOX.neonGreen}`,
          boxShadow: `inset -12px -14px 22px rgba(0, 0, 0, 0.45), ${XBOX.glow}`,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 26,
          borderRadius: "50%",
          background: "rgba(3, 17, 7, 0.95)",
          border: `2px solid ${XBOX.chartreuse}`,
          boxShadow: `inset 0 0 12px rgba(204, 255, 0, 0.28)`,
        }}
      />
      <div style={{ position: "absolute", top: 39, left: 10, right: 10, height: 2, background: "rgba(255, 255, 255, 0.45)", transform: "rotate(-18deg)" }} />
    </div>
  );
}

function ScreenHeader() {
  return (
    <>
      <FormatExportIcon />
      <h1 style={titleStyle}>Format & Export</h1>
      <div style={headerGlowLineStyle} />
    </>
  );
}

const FORMATS = [
  { id: "dpo", label: "DPO Preference", desc: "prompt / chosen / rejected", default: true },
  { id: "alpaca", label: "Alpaca", desc: "instruction tuning", default: false },
  { id: "chatml", label: "ChatML", desc: "conversation / multi-turn", default: false },
  { id: "openai_messages", label: "OpenAI Messages", desc: "modern chat models", default: false },
  { id: "completion", label: "Completion", desc: "pre-training / raw text", default: false },
  { id: "template_free", label: "Template-Free", desc: "custom formatting", default: false },
];

export function FormatExport({ onTrain }: { onTrain?: () => void }) {
  const [entries, setEntries] = useState<TrainingEntry[]>([]);
  const [fmt, setFmt] = useState("dpo");
  const [exported, setExported] = useState(false);
  const [exportResult, setExportResult] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    apiGet<TrainingEntry[]>("/api/training-entries")
      .then((data) => {
        setEntries(data);
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

  const accepted = useMemo(() => entries.filter((e) => e.status === "accepted" || e.status === "edited"), [entries]);
  const rejected = useMemo(() => entries.filter((e) => e.status === "rejected"), [entries]);

  const previewData = useMemo(() => {
    return accepted.slice(0, 5).map((e) => {
      if (fmt === "dpo") return { prompt: e.prompt, chosen: e.chosen, rejected: e.rejected };
      if (fmt === "alpaca") return { instruction: e.prompt, input: "", output: e.chosen };
      if (fmt === "chatml") return { messages: [{ role: "user", content: e.prompt }, { role: "assistant", content: e.chosen }] };
      if (fmt === "openai_messages") return { messages: [{ role: "user", content: e.prompt }, { role: "assistant", content: e.chosen }] };
      if (fmt === "completion") return { text: `${e.prompt}\n${e.chosen}` };
      return { prompt: e.prompt, response: e.chosen };
    });
  }, [accepted, fmt]);

  const doExport = async () => {
    setExporting(true);
    setExportError(null);
    try {
      const result = await apiPost("/api/format", {
        input: "data/mindforge.db",
        format: fmt,
        output: `data/training-data/${fmt}/train.jsonl`,
      });
      setExported(true);
      setExportResult(result);
    } catch (err: any) {
      setExportError(err?.message || String(err));
    } finally {
      setExporting(false);
    }
  };

  if (loading)
    return <LoadingState message="Loading training entries..." />;

  if (error)
    return (
      <div style={screenStyle}>
        <ScreenHeader />
        <ErrorState message={`Failed to load training entries: ${error}`} onRetry={load} />
      </div>
    );

  // Empty state: no training entries at all
  if (entries.length === 0) {
    return (
      <div style={screenStyle}>
        <ScreenHeader />
        <div className="panel" style={{ ...xboxPanelStyle, padding: 24 }}>
          <EmptyState
            icon="📦"
            title="No training data available"
            message="Run a probe and review entries to generate training data for export."
          />
        </div>
      </div>
    );
  }

  return (
    <div style={screenStyle}>
      <ScreenHeader />

      {/* Format Selection */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>
          Training Format
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
          {FORMATS.map((f) => {
            const isSelected = fmt === f.id;
            return (
            <motion.div
              key={f.id}
              role="button"
              tabIndex={0}
              aria-label={`${f.label} format${f.default ? " (default)" : ""}`}
              aria-pressed={fmt === f.id}
              whileHover={{ scale: 1.01 }}
              onClick={() => setFmt(f.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setFmt(f.id);
                }
              }}
              style={{
                padding: "10px 14px",
                borderRadius: 6,
                cursor: "pointer",
                color: isSelected ? XBOX.chartreuse : XBOX.primaryText,
                border: isSelected ? `1px solid ${XBOX.chartreuse}` : `1px solid ${XBOX.neonGreen}`,
                borderLeft: isSelected ? `3px solid ${XBOX.chartreuse}` : `1px solid ${XBOX.neonGreen}`,
                background: isSelected ? "rgba(204, 255, 0, 0.15)" : "transparent",
                boxShadow: isSelected ? XBOX.glow : "none",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, fontWeight: 600 }}>
                <span style={{ color: isSelected ? XBOX.chartreuse : XBOX.dimGreen }}>{fmt === f.id ? "●" : "○"}</span>
                {f.label}
                {f.default && <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: XBOX.chartreuse, color: "#001f08" }}>DEFAULT</span>}
              </div>
              <div style={{ fontSize: 11, color: XBOX.dimGreen, marginTop: 2, marginLeft: 20 }}>{f.desc}</div>
            </motion.div>
            );
          })}
        </div>
      </div>

      {/* Data Summary */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>
          Data Summary
        </h2>
        <div style={{ fontSize: 14, marginBottom: 12 }}>Total training pairs: <span style={{ color: XBOX.chartreuse, fontWeight: 700 }}>{accepted.length}</span></div>
        <div style={{ display: "flex", gap: 20, fontSize: 13 }}>
          <span style={{ color: "var(--success)" }}>✓ Correct (auto-approved): {accepted.length}</span>
          <span style={{ color: "var(--info)" }}>✎ Corrected (human-reviewed): {accepted.filter((e) => e.status === "edited").length}</span>
          <span style={{ color: "var(--error)" }}>✗ Rejected (discarded): {rejected.length}</span>
        </div>
      </div>

      {/* Output Preview */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>
          Output Preview ({fmt})
        </h2>
        <div className="mono" style={{ padding: 12, background: "rgba(0, 0, 0, 0.24)", borderRadius: 4, fontSize: 12, maxHeight: 300, overflowY: "auto", color: XBOX.primaryText }}>
          {previewData.length === 0 ? (
            <div style={{ color: XBOX.dimGreen }}>No data to preview. Run a probe and review entries first.</div>
          ) : (
            previewData.map((e, i) => (
              <div key={i} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: i < previewData.length - 1 ? "1px solid var(--surface)" : "none" }}>
                {JSON.stringify(e)}
              </div>
            ))
          )}
          {accepted.length > 5 && (
            <div style={{ color: XBOX.dimGreen, marginTop: 8 }}>... {accepted.length - 5} more rows</div>
          )}
        </div>
      </div>

      {/* Export Error */}
      {exportError && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 12, marginBottom: 16, borderLeft: "3px solid var(--error)" }}>
          <span style={{ color: "var(--error)", fontSize: 14, fontWeight: 600 }}>✗ Export failed: {exportError}</span>
        </div>
      )}

      {/* Export Result */}
      {exported && exportResult && !exportError && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="panel"
          style={{ ...xboxPanelStyle, padding: 16, marginBottom: 20, borderLeft: "3px solid var(--success)" }}
        >
          <div style={{ color: "var(--success)", fontSize: 14, fontWeight: 600 }}>✓ Export Complete</div>
          <div style={{ fontSize: 13, color: XBOX.dimGreen, marginTop: 4 }}>
            {exportResult.count} entries written to {exportResult.output}
          </div>
        </motion.div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 12 }}>
        <button className="btn-gold gold-glow" onClick={doExport} disabled={exporting} style={{ flex: 1, padding: 12, fontSize: 16, opacity: exporting ? 0.5 : 1 }}>
          {exporting ? "⏳ Exporting..." : "► Export Training Data"}
        </button>
        {exported && !exportError && (
          <button className="btn-gold" onClick={onTrain} style={{ flex: 1, padding: 12, fontSize: 16, background: "var(--accent-secondary)", color: "var(--bg)" }}>
            ► Continue to Train & Evaluate
          </button>
        )}
      </div>
    </div>
  );
}
