import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiGet, apiPost } from "../lib/api";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

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
    <div style={{ padding: 24, height: "100%", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Settings</h1>

      {/* Save feedback */}
      <AnimatePresence>
        {saveMessage && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="panel"
            style={{
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
              <span style={{ color: "var(--text-secondary)", fontSize: 12, marginLeft: 8 }}>
                — check that the server is running and the /api/settings endpoint exists
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Defaults indicator */}
      {usingDefaults && (
        <div className="panel" style={{ padding: 12, marginBottom: 16, borderLeft: "3px solid var(--warning)" }}>
          <span style={{ color: "var(--warning)", fontSize: 13 }}>
            ⚠ Using default values — server settings endpoint unavailable
          </span>
          {loadError && (
            <span style={{ color: "var(--text-dim)", fontSize: 11, marginLeft: 8 }}>{"(" + loadError + ")"}</span>
          )}
        </div>
      )}

      {/* Appearance */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h2 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          Appearance
        </h2>
        <div style={{ marginBottom: 12 }}>
          <label style={{ fontSize: 14, marginRight: 12 }}>Theme:</label>
          {["gold", "cyberpunk", "slate", "mono"].map((t) => (
            <button
              key={t}
              onClick={() => setSettings((s) => ({ ...s, theme: t }))}
              style={{
                padding: "4px 12px",
                marginRight: 8,
                borderRadius: 4,
                background: settings.theme === t ? "var(--accent)" : "var(--surface-raised)",
                color: settings.theme === t ? "var(--bg)" : "var(--text)",
                border: "1px solid var(--border)",
                cursor: "pointer",
                fontSize: 13,
              }}
            >
              {t}
            </button>
          ))}
        </div>
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: 14 }}>
            Sound Effects:{" "}
            <input
              type="checkbox"
              checked={settings.sound_effects ?? true}
              onChange={(e) => setSettings((s) => ({ ...s, sound_effects: e.target.checked }))}
            />
          </label>
        </div>
        <div>
          <label style={{ fontSize: 14 }}>
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
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h2 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
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
              width: 120,
              padding: 8,
              background: "var(--surface-raised)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              color: "var(--text)",
            }}
          />
          <span style={{ fontSize: 12, color: "var(--text-dim)", marginLeft: 8 }}>
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
              width: 120,
              padding: 8,
              background: "var(--surface-raised)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              color: "var(--text)",
            }}
          />
        </div>
      </div>

      {/* Data Paths (read-only info) */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h2 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
          Data Paths
        </h2>
        <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
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
