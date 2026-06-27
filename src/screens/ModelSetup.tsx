import { useState, useEffect, useCallback, type CSSProperties } from "react";
import { motion } from "framer-motion";
import { apiGet, type HardwareInfo, type ModelEntry, type ModelListResponse } from "../lib/api";
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
  overflowY: "auto",
  height: "100%",
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
  color: XBOX.neonGreen,
  fontSize: 14,
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

function ModelSetupIcon() {
  return (
    <div aria-hidden="true" style={decorativeIconStyle}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: "50%",
          background: `radial-gradient(circle at 28% 24%, #f4ffd9 0 8%, ${XBOX.chartreuse} 20%, #0a8f24 56%, #031b08 100%)`,
          border: `2px solid ${XBOX.neonGreen}`,
          boxShadow: `inset -14px -18px 26px rgba(0, 0, 0, 0.55), ${XBOX.glow}`,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "grid",
          placeItems: "center",
          color: "#001f08",
          fontFamily: "'Arial Black', Impact, sans-serif",
          fontSize: 48,
          fontWeight: 900,
          textShadow: `0 0 10px ${XBOX.chartreuse}`,
        }}
      >
        X
      </div>
    </div>
  );
}

function ScreenHeader() {
  return (
    <>
      <ModelSetupIcon />
      <h1 style={titleStyle}>Model Setup</h1>
      <div style={headerGlowLineStyle} />
    </>
  );
}

export function ModelSetup({ onContinue, onSelectModel }: { onContinue?: () => void; onSelectModel?: (model: string) => void }) {
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [cloudModels, setCloudModels] = useState<ModelEntry[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      apiGet<HardwareInfo>("/api/hardware"),
      apiGet<ModelListResponse>("/api/models"),
    ])
      .then(([h, m]) => {
        setHw(h);
        setModels(m.local || m.local_models || []);
        setCloudModels(m.cloud || m.cloud_models || []);
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
      <div style={screenStyle}>
        <ScreenHeader />
        <ErrorState message={`Failed to load hardware/models: ${error}`} onRetry={load} />
      </div>
    );

  // Compute usable memory and tier if the API doesn't provide them
  const usableMem = hw?.usable_memory_gb ?? (hw?.memory_gb ? hw.memory_gb * 0.75 : 0);
  const tier = hw?.tier ?? (usableMem >= 96 ? "S" : usableMem >= 64 ? "A" : usableMem >= 32 ? "B" : usableMem >= 16 ? "C" : usableMem >= 8 ? "D" : "E");
  const memPct = hw ? (usableMem / (hw.memory_gb || 1)) * 100 : 0;
  const modelLabel = hw?.model || hw?.model_name || "Unknown";

  return (
    <div style={screenStyle}>
      <ScreenHeader />

      {/* Hardware Detected */}
      {hw && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20, borderLeft: "3px solid var(--success)" }}>
          <h2 style={sectionHeadingStyle}>
            ✓ Hardware Detected
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 14, marginBottom: 12 }}>
            <div>Chip: <span style={{ color: XBOX.chartreuse }}>{hw.chip}</span></div>
            <div>Model: <span style={{ color: XBOX.chartreuse }}>{modelLabel}</span></div>
            <div>Memory: <span style={{ color: XBOX.chartreuse }}>{hw.memory_gb?.toFixed(1)} GB</span></div>
            <div>Usable: <span style={{ color: XBOX.chartreuse }}>{usableMem.toFixed(1)} GB</span></div>
            <div>Tier: <span style={{ color: XBOX.chartreuse }}>{tier}</span></div>
          </div>
          <div className="progress-bar" style={{ height: 10 }}>
            <div className="progress-fill" style={{ width: `${memPct}%`, background: "var(--success)" }} />
          </div>
          <div style={{ fontSize: 11, color: XBOX.dimGreen, marginTop: 4 }}>
            {usableMem.toFixed(0)} / {hw.memory_gb?.toFixed(0)} GB
          </div>
        </div>
      )}

      {/* Local Models */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>
          🖥 Local (MLX)
        </h2>
        {models.length === 0 ? (
          <div style={{ color: XBOX.dimGreen, fontSize: 13 }}>No local models detected. Run <code>mindforge detect</code> to scan.</div>
        ) : (
          models.map((m, i) => {
            const repo = m.repo || m.id || m.name;
            const canRun = m.can_run ?? m.available ?? true;
            const isSelected = selected === repo;
            return (
            <motion.div
              key={i}
              role={canRun ? "button" : undefined}
              tabIndex={canRun ? 0 : undefined}
              aria-label={`Select model ${m.name}`}
              aria-pressed={selected === repo}
              whileHover={canRun ? { scale: 1.01 } : {}}
              onClick={() => { if (canRun) { setSelected(repo); onSelectModel?.(repo); } }}
              onKeyDown={(e) => {
                if (canRun && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  setSelected(repo);
                  onSelectModel?.(repo);
                }
              }}
              style={{
                padding: "10px 12px",
                marginBottom: 4,
                borderRadius: 6,
                cursor: canRun ? "pointer" : "not-allowed",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                boxSizing: "border-box",
                color: canRun ? XBOX.primaryText : XBOX.dimGreen,
                background: isSelected ? "rgba(204, 255, 0, 0.15)" : "transparent",
                border: isSelected ? `1px solid ${XBOX.chartreuse}` : "1px solid transparent",
                borderLeft: isSelected ? `3px solid ${XBOX.chartreuse}` : "3px solid transparent",
                boxShadow: isSelected ? XBOX.glow : "none",
                opacity: canRun ? 1 : 0.4,
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ color: isSelected ? XBOX.chartreuse : XBOX.dimGreen }}>
                  {selected === repo ? "●" : "○"}
                </span>
                {m.name}
                {m.badge && (
                  <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 3, background: "var(--accent-secondary)", color: "var(--bg)" }}>
                    {m.badge}
                  </span>
                )}
              </span>
              <span style={{ color: XBOX.dimGreen, fontSize: 12 }}>
                ~{m.size_gb} GB · {m.tier}
              </span>
            </motion.div>
            );
          })
        )}
      </div>

      {/* Cloud Models */}
      {cloudModels.length > 0 && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
          <h2 style={sectionHeadingStyle}>
            ☁ Cloud (API)
          </h2>
          {cloudModels.map((m, i) => {
            const isSelected = selected === m.repo;
            return (
            <div
              key={i}
              role="button"
              tabIndex={0}
              aria-label={`Select model ${m.name}`}
              aria-pressed={selected === m.repo}
              onClick={() => { setSelected(m.repo); onSelectModel?.(m.repo); }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setSelected(m.repo);
                  onSelectModel?.(m.repo);
                }
              }}
              style={{
                padding: "10px 12px",
                marginBottom: 4,
                borderRadius: 6,
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                boxSizing: "border-box",
                color: XBOX.primaryText,
                background: isSelected ? "rgba(204, 255, 0, 0.15)" : "transparent",
                border: isSelected ? `1px solid ${XBOX.chartreuse}` : "1px solid transparent",
                borderLeft: isSelected ? `3px solid ${XBOX.chartreuse}` : "3px solid transparent",
                boxShadow: isSelected ? XBOX.glow : "none",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ color: isSelected ? XBOX.chartreuse : XBOX.dimGreen }}>
                  {selected === m.repo ? "●" : "○"}
                </span>
                {m.name}
              </span>
              <span style={{ color: XBOX.dimGreen, fontSize: 12 }}>{m.type}</span>
            </div>
            );
          })}
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
        <div style={{ marginTop: 8, fontSize: 13, color: XBOX.dimGreen }}>
          Selected: <span style={{ color: XBOX.chartreuse }}>{selected}</span>
        </div>
      )}
    </div>
  );
}
