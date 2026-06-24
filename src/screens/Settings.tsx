import { useState } from "react";

export function Settings() {
  const [theme, setTheme] = useState("gold");
  const [sound, setSound] = useState(true);
  const [animations, setAnimations] = useState(true);

  return (
    <div style={{ padding: 24, height: "100vh", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Settings</h1>
      
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12 }}>Appearance</h3>
        <div style={{ marginBottom: 12 }}>
          <label style={{ fontSize: 14, marginRight: 12 }}>Theme:</label>
          {["gold", "cyberpunk", "slate", "mono"].map(t => (
            <button key={t} onClick={() => setTheme(t)} style={{ padding: "4px 12px", marginRight: 8, borderRadius: 4,
              background: theme === t ? "var(--accent)" : "var(--surface-raised)",
              color: theme === t ? "var(--bg)" : "var(--text)", border: "1px solid var(--border)", cursor: "pointer", fontSize: 13 }}>
              {t}
            </button>
          ))}
        </div>
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: 14 }}>Sound Effects: </label>
          <input type="checkbox" checked={sound} onChange={e => setSound(e.target.checked)} />
        </div>
        <div>
          <label style={{ fontSize: 14 }}>Animations: </label>
          <input type="checkbox" checked={animations} onChange={e => setAnimations(e.target.checked)} />
        </div>
      </div>

      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12 }}>Data Paths</h3>
        <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
          <div>Data: ~/mindforge-data/</div>
          <div>Output: ~/mindforge-data/training-data/</div>
          <div>Model cache: ~/.cache/huggingface/</div>
        </div>
      </div>

      <div className="panel" style={{ padding: 20 }}>
        <h3 style={{ marginBottom: 12 }}>Probing Config</h3>
        <div style={{ fontSize: 14 }}>Auto-approve threshold: 0.7</div>
        <div style={{ fontSize: 14 }}>Max questions per subject: 25</div>
      </div>
    </div>
  );
}
