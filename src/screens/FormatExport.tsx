import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { apiGet, apiPost, type TrainingEntry } from "../lib/api";

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

  useEffect(() => {
    apiGet<TrainingEntry[]>("/api/training-entries").then(setEntries).catch(() => {});
  }, []);

  const accepted = useMemo(() => entries.filter((e) => e.status === "accepted" || e.status === "edited"), [entries]);
  const rejected = useMemo(() => entries.filter((e) => e.status === "rejected"), [entries]);
  const pending = useMemo(() => entries.filter((e) => e.status === "pending"), [entries]);

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
    try {
      const result = await apiPost("/api/format", {
        input: "data/mindforge.db",
        format: fmt,
        output: `data/training-data/${fmt}/train.jsonl`,
      });
      setExported(true);
      setExportResult(result);
    } catch {}
  };

  return (
    <div style={{ padding: 24, height: "100%", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Format & Export</h1>

      {/* Format Selection */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          Training Format
        </h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
          {FORMATS.map((f) => (
            <motion.div
              key={f.id}
              whileHover={{ scale: 1.01 }}
              onClick={() => setFmt(f.id)}
              style={{
                padding: "10px 14px",
                borderRadius: 6,
                cursor: "pointer",
                border: fmt === f.id ? "1px solid var(--accent)" : "1px solid var(--border)",
                background: fmt === f.id ? "var(--surface-raised)" : "transparent",
                boxShadow: fmt === f.id ? "0 0 12px var(--accent-glow)" : "none",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, fontWeight: 600 }}>
                <span style={{ color: fmt === f.id ? "var(--accent)" : "var(--text-dim)" }}>{fmt === f.id ? "●" : "○"}</span>
                {f.label}
                {f.default && <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: "var(--accent)", color: "var(--bg)" }}>DEFAULT</span>}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 2, marginLeft: 20 }}>{f.desc}</div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Data Summary */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          Data Summary
        </h3>
        <div style={{ fontSize: 14, marginBottom: 12 }}>Total training pairs: <span style={{ color: "var(--accent)", fontWeight: 700 }}>{accepted.length}</span></div>
        <div style={{ display: "flex", gap: 20, fontSize: 13 }}>
          <span style={{ color: "var(--success)" }}>✓ Correct (auto-approved): {accepted.length}</span>
          <span style={{ color: "var(--info)" }}>✎ Corrected (human-reviewed): {accepted.filter((e) => e.status === "edited").length}</span>
          <span style={{ color: "var(--error)" }}>✗ Rejected (discarded): {rejected.length}</span>
        </div>
      </div>

      {/* Output Preview */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          Output Preview ({fmt})
        </h3>
        <div className="mono" style={{ padding: 12, background: "var(--surface-raised)", borderRadius: 4, fontSize: 12, maxHeight: 300, overflowY: "auto" }}>
          {previewData.length === 0 ? (
            <div style={{ color: "var(--text-dim)" }}>No data to preview. Run a probe and review entries first.</div>
          ) : (
            previewData.map((e, i) => (
              <div key={i} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: i < previewData.length - 1 ? "1px solid var(--surface)" : "none" }}>
                {JSON.stringify(e)}
              </div>
            ))
          )}
          {accepted.length > 5 && (
            <div style={{ color: "var(--text-dim)", marginTop: 8 }}>... {accepted.length - 5} more rows</div>
          )}
        </div>
      </div>

      {/* Export Result */}
      {exported && exportResult && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="panel"
          style={{ padding: 16, marginBottom: 20, borderLeft: "3px solid var(--success)" }}
        >
          <div style={{ color: "var(--success)", fontSize: 14, fontWeight: 600 }}>✓ Export Complete</div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>
            {exportResult.count} entries written to {exportResult.output}
          </div>
        </motion.div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 12 }}>
        <button className="btn-gold gold-glow" onClick={doExport} style={{ flex: 1, padding: 12, fontSize: 16 }}>
          ► Export Training Data
        </button>
        {exported && (
          <button className="btn-gold" onClick={onTrain} style={{ flex: 1, padding: 12, fontSize: 16, background: "var(--accent-secondary)", color: "var(--bg)" }}>
            ► Continue to Train & Evaluate
          </button>
        )}
      </div>
    </div>
  );
}
