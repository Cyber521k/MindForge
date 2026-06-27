import { useState, useEffect, useCallback, type CSSProperties } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiGet, apiPost } from "../lib/api";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

const XBOX = {
  primaryText: "#FFF8DC",
  neonGreen: "#33FF33",
  chartreuse: "#CCFF00",
  dimGreen: "#1A3A1A",
  glow: "var(--xbox-glow, 0 0 18px rgba(51, 255, 51, 0.45))",
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
  background: "linear-gradient(90deg, transparent, #33FF33, transparent)",
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

const fieldStyle: CSSProperties = {
  padding: 8,
  background: "rgba(0, 0, 0, 0.24)",
  border: `1px solid ${XBOX.neonGreen}`,
  borderRadius: 4,
  color: XBOX.primaryText,
};

const activeItemStyle: CSSProperties = {
  background: "rgba(204, 255, 0, 0.15)",
  borderLeft: `3px solid ${XBOX.chartreuse}`,
  boxShadow: XBOX.glow,
  color: XBOX.chartreuse,
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

function SettingsIcon() {
  return (
    <div aria-hidden="true" style={decorativeIconStyle}>
      <div
        style={{
          position: "absolute",
          inset: 8,
          background: `conic-gradient(from 10deg, ${XBOX.neonGreen} 0 9%, transparent 9% 16%, ${XBOX.chartreuse} 16% 25%, transparent 25% 34%, ${XBOX.neonGreen} 34% 43%, transparent 43% 50%, ${XBOX.chartreuse} 50% 59%, transparent 59% 66%, ${XBOX.neonGreen} 66% 75%, transparent 75% 84%, ${XBOX.chartreuse} 84% 93%, transparent 93% 100%)`,
          clipPath: "polygon(50% 0, 61% 18%, 82% 10%, 90% 30%, 72% 42%, 100% 50%, 72% 58%, 90% 70%, 82% 90%, 61% 82%, 50% 100%, 39% 82%, 18% 90%, 10% 70%, 28% 58%, 0 50%, 28% 42%, 10% 30%, 18% 10%, 39% 18%)",
          boxShadow: XBOX.glow,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 22,
          borderRadius: "50%",
          background: "rgba(10, 26, 10, 0.96)",
          border: `2px solid ${XBOX.neonGreen}`,
          boxShadow: `inset 0 0 16px rgba(51, 255, 51, 0.24), ${XBOX.glow}`,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 32,
          borderRadius: "50%",
          background: XBOX.chartreuse,
          boxShadow: `0 0 14px ${XBOX.chartreuse}`,
        }}
      />
    </div>
  );
}

function ScreenHeader() {
  return (
    <>
      <SettingsIcon />
      <h1 style={titleStyle}>Settings</h1>
      <div style={headerGlowLineStyle} />
    </>
  );
}

interface SettingsData {
  theme?: string;
  sound_effects?: boolean;
  animations?: boolean;
  auto_approve_threshold?: number;
  max_questions_per_subject?: number;
  [key: string]: any;
}

const DEFAULTS: SettingsData = {
  theme: "gold",
  sound_effects: true,
  animations: true,
  auto_approve_threshold: 0.7,
  max_questions_per_subject: 25,
};

export function Settings() {
  const [settings, setSettings] = useState<SettingsData>(DEFAULTS);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [usingDefaults, setUsingDefaults] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setLoadError(null);
    apiGet<SettingsData>("/api/settings")
      .then((data) => {
        setSettings({ ...DEFAULTS, ...data });
        setUsingDefaults(false);
        setLoading(false);
      })
      .catch((err) => {
        // Endpoint may not exist yet — use defaults
        setSettings(DEFAULTS);
        setUsingDefaults(true);
        setLoadError(err?.message || String(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Auto-dismiss save message after 3 seconds on success
  useEffect(() => {
    if (saveMessage?.type === "success") {
      const timer = setTimeout(() => setSaveMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [saveMessage]);

  const save = async () => {
    setSaving(true);
    setSaveMessage(null);
    try {
      await apiPost("/api/settings", settings);
      setSaveMessage({ type: "success", text: "✓ Settings saved" });
      setUsingDefaults(false);
      setLoadError(null);
    } catch (err: any) {
      setSaveMessage({
        type: "error",
        text: `✗ Failed to save: ${err?.message || String(err)}`,
      });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <LoadingState message="Loading settings..." />;
  }

  if (loadError && usingDefaults) {
    // Show the settings with defaults but note the endpoint is unavailable
    // Still allow editing and attempting to save
  }

  return (
    <div style={screenStyle}>
      <ScreenHeader />

      {/* Save feedback */}
      <AnimatePresence>
        {saveMessage && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="panel"
            style={{
              ...xboxPanelStyle,
              padding: 12,
              marginBottom: 16,
              borderLeft: `3px solid ${saveMessage.type === "success" ? "var(--success)" : "var(--error)"}`,
              overflow: "hidden",
            }}
          >
            <span style={{ color: saveMessage.type === "success" ? "var(--success)" : "var(--error)", fontSize: 14, fontWeight: 600 }}>
              {saveMessage.text}
            </span>
            {saveMessage.type === "error" && (
              <span style={{ color: XBOX.dimGreen, fontSize: 12, marginLeft: 8 }}>
                — check that the server is running and the /api/settings endpoint exists
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Defaults indicator */}
      {usingDefaults && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 12, marginBottom: 16, borderLeft: "3px solid var(--warning)" }}>
          <span style={{ color: "var(--warning)", fontSize: 13 }}>
            ⚠ Using default values — server settings endpoint unavailable
          </span>
          {loadError && (
            <span style={{ color: XBOX.dimGreen, fontSize: 11, marginLeft: 8 }}>{"(" + loadError + ")"}</span>
          )}
        </div>
      )}

      {/* Appearance */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>
          Appearance
        </h2>
        <div style={{ marginBottom: 12 }}>
          <label style={{ fontSize: 14, marginRight: 12 }}>Theme:</label>
          {["gold", "cyberpunk", "slate", "mono"].map((t) => {
            const isSelected = settings.theme === t;
            return (
              <button
                key={t}
                onClick={() => setSettings((s) => ({ ...s, theme: t }))}
                style={{
                  padding: "4px 12px",
                  marginRight: 8,
                  borderRadius: 4,
                  background: isSelected ? activeItemStyle.background : "rgba(10, 26, 10, 0.7)",
                  color: isSelected ? XBOX.chartreuse : XBOX.primaryText,
                  border: isSelected ? `1px solid ${XBOX.chartreuse}` : `1px solid ${XBOX.neonGreen}`,
                  borderLeft: isSelected ? activeItemStyle.borderLeft : `1px solid ${XBOX.neonGreen}`,
                  boxShadow: isSelected ? XBOX.glow : "none",
                  cursor: "pointer",
                  fontSize: 13,
                }}
              >
                {t}
              </button>
            );
          })}
        </div>
        <div style={{ marginBottom: 8 }}>
          <label
            style={{
              fontSize: 14,
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 8px",
              ...(settings.sound_effects ? activeItemStyle : {}),
            }}
          >
            Sound Effects:{" "}
            <input
              type="checkbox"
              checked={settings.sound_effects ?? true}
              onChange={(e) => setSettings((s) => ({ ...s, sound_effects: e.target.checked }))}
            />
          </label>
        </div>
        <div>
          <label
            style={{
              fontSize: 14,
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 8px",
              ...(settings.animations ? activeItemStyle : {}),
            }}
          >
            Animations:{" "}
            <input
              type="checkbox"
              checked={settings.animations ?? true}
              onChange={(e) => setSettings((s) => ({ ...s, animations: e.target.checked }))}
            />
          </label>
        </div>
      </div>

      {/* Probing Config */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>
          Probing Config
        </h2>
        <div style={{ marginBottom: 12 }}>
          <label style={{ fontSize: 14, display: "block", marginBottom: 4 }}>
            Auto-approve threshold:
          </label>
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={settings.auto_approve_threshold ?? 0.7}
            onChange={(e) =>
              setSettings((s) => ({ ...s, auto_approve_threshold: parseFloat(e.target.value) || 0 }))
            }
            style={{
              ...fieldStyle,
              width: 120,
            }}
          />
          <span style={{ fontSize: 12, color: XBOX.dimGreen, marginLeft: 8 }}>
            (0.0–1.0, responses above this go to auto-approved)
          </span>
        </div>
        <div>
          <label style={{ fontSize: 14, display: "block", marginBottom: 4 }}>
            Max questions per subject:
          </label>
          <input
            type="number"
            min="1"
            max="500"
            value={settings.max_questions_per_subject ?? 25}
            onChange={(e) =>
              setSettings((s) => ({ ...s, max_questions_per_subject: parseInt(e.target.value) || 1 }))
            }
            style={{
              ...fieldStyle,
              width: 120,
            }}
          />
        </div>
      </div>

      {/* Data Paths (read-only info) */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>
          Data Paths
        </h2>
        <div style={{ fontSize: 13, color: XBOX.dimGreen }}>
          <div>Data: ~/mindforge-data/</div>
          <div>Output: ~/mindforge-data/training-data/</div>
          <div>Model cache: ~/.cache/huggingface/</div>
        </div>
      </div>

      {/* Save Button */}
      <button
        className="btn-gold gold-glow"
        onClick={save}
        disabled={saving}
        style={{ width: "100%", padding: 14, fontSize: 16, opacity: saving ? 0.5 : 1 }}
      >
        {saving ? "⏳ Saving..." : "✓ Save Settings"}
      </button>
    </div>
  );
}
