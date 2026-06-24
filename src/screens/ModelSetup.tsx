import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { apiGet, type HardwareInfo, type ModelEntry, type ModelListResponse } from "../lib/api";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

export function ModelSetup({ onContinue }: { onContinue?: () => void }) {
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [cloudModels, setCloudModels] = useState<ModelEntry[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    apiGet<HardwareInfo>("/api/hardware")
      .then((h) => {
        setHw(h);
        return apiGet<ModelListResponse>("/api/models");
      })
      .then((m) => {
        setModels(m.local || []);
        setCloudModels(m.cloud || []);
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

  if (loading)
    return <LoadingState message="⚕ Detecting hardware..." />;

  if (error)
    return (
      <div style={{ padding: 24 }}>
        <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Model Setup</h1>
        <ErrorState message={`Failed to load hardware/models: ${error}`} onRetry={load} />
      </div>
    );

  const memPct = hw ? (hw.usable_memory_gb / hw.memory_gb) * 100 : 0;

  return (
    <div style={{ padding: 24, overflowY: "auto", height: "100%" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Model Setup</h1>

      {/* Hardware Detected */}
      {hw && (
        <div className="panel" style={{ padding: 20, marginBottom: 20, borderLeft: "3px solid var(--success)" }}>
          <h3 style={{ marginBottom: 12, color: "var(--success)", fontSize: 14, textTransform: "uppercase", letterSpacing: 1 }}>
            ✓ Hardware Detected
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 14, marginBottom: 12 }}>
            <div>Chip: <span style={{ color: "var(--accent)" }}>{hw.chip}</span></div>
            <div>Model: <span style={{ color: "var(--accent)" }}>{hw.model}</span></div>
            <div>Memory: <span style={{ color: "var(--accent)" }}>{hw.memory_gb?.toFixed(1)} GB</span></div>
            <div>Usable: <span style={{ color: "var(--accent)" }}>{hw.usable_memory_gb?.toFixed(1)} GB</span></div>
            <div>Tier: <span style={{ color: "var(--accent)" }}>{hw.tier || "C"}</span></div>
          </div>
          <div className="progress-bar" style={{ height: 10 }}>
            <div className="progress-fill" style={{ width: `${memPct}%`, background: "var(--success)" }} />
          </div>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
            {hw.usable_memory_gb?.toFixed(0)} / {hw.memory_gb?.toFixed(0)} GB
          </div>
        </div>
      )}

      {/* Local Models */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, color: "var(--accent-secondary)", fontSize: 14, textTransform: "uppercase", letterSpacing: 1 }}>
          🖥 Local (MLX)
        </h3>
        {models.length === 0 ? (
          <div style={{ color: "var(--text-dim)", fontSize: 13 }}>No local models detected. Run <code>mindforge detect</code> to scan.</div>
        ) : (
          models.map((m, i) => (
            <motion.div
              key={i}
              whileHover={m.can_run ? { scale: 1.01 } : {}}
              onClick={() => m.can_run && setSelected(m.repo)}
              style={{
                padding: "10px 12px",
                marginBottom: 4,
                borderRadius: 6,
                cursor: m.can_run ? "pointer" : "not-allowed",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                background: selected === m.repo ? "var(--surface-raised)" : "transparent",
                border: selected === m.repo ? "1px solid var(--accent)" : "1px solid transparent",
                opacity: m.can_run ? 1 : 0.4,
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ color: selected === m.repo ? "var(--accent)" : "var(--text-dim)" }}>
                  {selected === m.repo ? "●" : "○"}
                </span>
                {m.name}
                {m.badge && (
                  <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 3, background: "var(--accent-secondary)", color: "var(--bg)" }}>
                    {m.badge}
                  </span>
                )}
              </span>
              <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                ~{m.size_gb} GB · {m.tier}
              </span>
            </motion.div>
          ))
        )}
      </div>

      {/* Cloud Models */}
      {cloudModels.length > 0 && (
        <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
          <h3 style={{ marginBottom: 12, color: "var(--info)", fontSize: 14, textTransform: "uppercase", letterSpacing: 1 }}>
            ☁ Cloud (API)
          </h3>
          {cloudModels.map((m, i) => (
            <div
              key={i}
              onClick={() => setSelected(m.repo)}
              style={{
                padding: "10px 12px",
                marginBottom: 4,
                borderRadius: 6,
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                background: selected === m.repo ? "var(--surface-raised)" : "transparent",
                border: selected === m.repo ? "1px solid var(--accent)" : "1px solid transparent",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ color: selected === m.repo ? "var(--accent)" : "var(--text-dim)" }}>
                  {selected === m.repo ? "●" : "○"}
                </span>
                {m.name}
              </span>
              <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>{m.type}</span>
            </div>
          ))}
        </div>
      )}

      {/* Continue Button */}
      {selected && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="btn-gold gold-glow"
          onClick={onContinue}
          style={{ width: "100%", padding: 14, fontSize: 16 }}
        >
          ► Continue to Domain Setup
        </motion.button>
      )}

      {selected && (
        <div style={{ marginTop: 8, fontSize: 13, color: "var(--text-secondary)" }}>
          Selected: <span style={{ color: "var(--accent)" }}>{selected}</span>
        </div>
      )}
    </div>
  );
}
