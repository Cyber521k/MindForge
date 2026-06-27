import { useState, useEffect, type CSSProperties } from "react";
import { motion } from "framer-motion";
import { apiPost } from "../lib/api";
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

const sectionHeadingStyle: CSSProperties = {
  marginBottom: 12,
  fontSize: 14,
  color: XBOX.neonGreen,
  textTransform: "uppercase",
  letterSpacing: 0,
};

const fieldLabelStyle: CSSProperties = {
  fontSize: 12,
  color: XBOX.dimGreen,
  display: "block",
  marginBottom: 4,
};

const fieldStyle: CSSProperties = {
  width: "100%",
  padding: 8,
  background: "rgba(0, 0, 0, 0.24)",
  border: `1px solid ${XBOX.neonGreen}`,
  borderRadius: 4,
  color: XBOX.primaryText,
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

function TrainEvaluateIcon() {
  return (
    <div aria-hidden="true" style={decorativeIconStyle}>
      {[0, 14, 28].map((offset, index) => (
        <div
          key={offset}
          style={{
            position: "absolute",
            inset: offset,
            borderRadius: "50%",
            border: `${index === 0 ? 3 : 2}px solid ${index === 1 ? XBOX.chartreuse : XBOX.neonGreen}`,
            boxShadow: index === 0 ? XBOX.glow : `inset 0 0 10px rgba(0, 255, 65, 0.18)`,
            background: index === 2 ? `radial-gradient(circle, ${XBOX.chartreuse} 0 20%, transparent 22%)` : "transparent",
          }}
        />
      ))}
      <div style={{ position: "absolute", left: 38, top: 2, bottom: 2, width: 2, background: XBOX.neonGreen, opacity: 0.65 }} />
      <div style={{ position: "absolute", top: 38, left: 2, right: 2, height: 2, background: XBOX.neonGreen, opacity: 0.65 }} />
    </div>
  );
}

function ScreenHeader() {
  return (
    <>
      <TrainEvaluateIcon />
      <h1 style={titleStyle}>Train & Evaluate</h1>
      <div style={headerGlowLineStyle} />
    </>
  );
}

interface LossPoint {
  iter: number;
  loss: number;
}

interface EvalResult {
  model: string;
  task: string;
  score: number;
  [key: string]: any;
}

export function TrainEvaluate() {
  const [model, setModel] = useState("mlx-community/Llama-3.2-3B-Instruct-4bit");
  const [data, setData] = useState("data/training-data/dpo/");
  const [mode, setMode] = useState("dpo");
  const [adapter, setAdapter] = useState("lora");
  const [iters, setIters] = useState(1000);
  const [batchSize, setBatchSize] = useState(4);
  const [learningRate, setLearningRate] = useState(1e-5);
  const [beta, setBeta] = useState(0.1);
  const [running, setRunning] = useState(false);
  const [currentIter, setCurrentIter] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [jobId, setJobId] = useState("");
  const [trainError, setTrainError] = useState<string | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const { latest } = useWebSocket();

  // WebSocket updates
  useEffect(() => {
    if (!latest) return;
    if (latest.type === "progress" && latest.job_id === jobId) {
      if (latest.iteration !== undefined) setCurrentIter(latest.iteration);
      if (latest.loss !== undefined) {
        setCurrentLoss(latest.loss);
        setLossHistory((prev) => [...prev, { iter: latest.iteration || currentIter, loss: latest.loss }].slice(-200));
      }
    }
    if (latest.type === "job_complete" && latest.job_id === jobId) {
      setRunning(false);
      setCurrentIter(iters);
    }
    if (latest.type === "job_failed" && latest.job_id === jobId) {
      setRunning(false);
      setTrainError(latest.error || "Training job failed");
    }
  }, [latest, jobId, iters, currentIter]);

  const startTrain = async () => {
    setRunning(true);
    setCurrentIter(0);
    setLossHistory([]);
    setEvalResult(null);
    setTrainError(null);
    try {
      const res = await apiPost("/api/train", { model, data, mode, iters, batch_size: batchSize, learning_rate: learningRate, beta });
      setJobId(res.job_id);
    } catch (err: any) {
      setRunning(false);
      setTrainError(err?.message || String(err));
    }
  };

  const startEval = async () => {
    setEvaluating(true);
    setEvalError(null);
    try {
      const res = await apiPost("/api/evaluate", { model, tasks: "mmlu_stem", num_fewshot: 5 });
      setEvalResult(res);
    } catch (err: any) {
      setEvalError(err?.message || String(err));
    } finally {
      setEvaluating(false);
    }
  };

  const pct = iters > 0 ? (currentIter / iters) * 100 : 0;
  const lossDelta = lossHistory.length >= 2 ? lossHistory[lossHistory.length - 1].loss - lossHistory[lossHistory.length - 2].loss : 0;

  // Loss curve SVG points
  const maxLoss = Math.max(...lossHistory.map((p) => p.loss), 1);
  const minLoss = Math.min(...lossHistory.map((p) => p.loss), 0);
  const lossRange = maxLoss - minLoss || 1;

  return (
    <div style={screenStyle}>
      <ScreenHeader />

      {/* Training Error */}
      {trainError && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 12, marginBottom: 16, borderLeft: "3px solid var(--error)" }}>
          <span style={{ color: "var(--error)", fontSize: 14, fontWeight: 600 }}>✗ Training failed: {trainError}</span>
          <button
            onClick={() => { setTrainError(null); startTrain(); }}
            style={{ marginLeft: 12, padding: "4px 12px", fontSize: 12, background: "rgba(10, 26, 10, 0.7)", color: XBOX.primaryText, border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4, cursor: "pointer" }}
          >
            ↻ Retry
          </button>
        </div>
      )}

      {/* Base Model Selection */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>Base Model</h2>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: XBOX.chartreuse }}>●</span>
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            aria-label="Base model"
            style={{ ...fieldStyle, flex: 1 }}
          />
        </div>
      </div>

      {/* Training Config */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
        <h2 style={sectionHeadingStyle}>Training Config</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
          <div>
            <label style={fieldLabelStyle}>Method</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)} style={fieldStyle}>
              <option value="dpo">DPO (preference)</option>
              <option value="sft">SFT (supervised)</option>
              <option value="orpo">ORPO</option>
            </select>
          </div>
          <div>
            <label style={fieldLabelStyle}>Adapter</label>
            <select value={adapter} onChange={(e) => setAdapter(e.target.value)} style={fieldStyle}>
              <option value="lora">LoRA</option>
              <option value="dora">DoRA</option>
              <option value="full">Full</option>
            </select>
          </div>
          <div>
            <label style={fieldLabelStyle}>Data Path</label>
            <input value={data} onChange={(e) => setData(e.target.value)} style={fieldStyle} />
          </div>
          <div>
            <label style={fieldLabelStyle}>Iterations</label>
            <input type="number" value={iters} onChange={(e) => setIters(+e.target.value)} style={fieldStyle} />
          </div>
          <div>
            <label style={fieldLabelStyle}>Batch Size</label>
            <input type="number" value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} style={fieldStyle} />
          </div>
          <div>
            <label style={fieldLabelStyle}>Learning Rate</label>
            <input type="number" step="0.00001" value={learningRate} onChange={(e) => setLearningRate(+e.target.value)} style={fieldStyle} />
          </div>
          {mode === "dpo" && (
            <div>
              <label style={fieldLabelStyle}>Beta (DPO)</label>
              <input type="number" step="0.01" value={beta} onChange={(e) => setBeta(+e.target.value)} style={fieldStyle} />
            </div>
          )}
        </div>
        <button
          className="btn-gold gold-glow"
          onClick={startTrain}
          disabled={running}
          style={{ width: "100%", padding: 12, fontSize: 16, opacity: running ? 0.5 : 1, background: running ? "rgba(204, 255, 0, 0.15)" : undefined, borderLeft: running ? `3px solid ${XBOX.chartreuse}` : undefined, boxShadow: running ? XBOX.glow : undefined }}
        >
          {running ? "⏸ Training..." : "► Start Fine-Tuning"}
        </button>
      </div>

      {/* Training Progress */}
      {(running || currentIter > 0) && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 20, marginBottom: 20 }}>
          <h2 style={sectionHeadingStyle}>Training Progress</h2>
          <div style={{ fontSize: 14, marginBottom: 8 }}>
            Iteration <span style={{ color: XBOX.chartreuse }}>{currentIter}</span> / {iters}
          </div>
          <div className="progress-bar" style={{ height: 12, marginBottom: 8 }}>
            <motion.div className="progress-fill" animate={{ width: `${pct}%` }} style={{ width: `${pct}%` }} />
          </div>
          <div style={{ fontSize: 13, color: XBOX.dimGreen, marginBottom: 12 }}>
            {pct.toFixed(1)}% · Loss: <span style={{ color: XBOX.chartreuse }}>{currentLoss.toFixed(4)}</span>
            {lossDelta !== 0 && (
              <span style={{ color: lossDelta < 0 ? "var(--success)" : "var(--error)", marginLeft: 8 }}>
                ({lossDelta < 0 ? "↓" : "↑"} {Math.abs(lossDelta).toFixed(4)})
              </span>
            )}
          </div>

          {/* Loss Curve */}
          {lossHistory.length > 1 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 12, color: XBOX.dimGreen, marginBottom: 4 }}>Loss Curve</div>
              <svg width="100%" height="120" style={{ background: "rgba(0, 0, 0, 0.24)", borderRadius: 4 }}>
                <polyline
                  fill="none"
                  stroke={XBOX.chartreuse}
                  strokeWidth="2"
                  points={lossHistory
                    .map((p, i) => {
                      const x = (i / Math.max(lossHistory.length - 1, 1)) * 100;
                      const y = 120 - ((p.loss - minLoss) / lossRange) * 100 - 10;
                      return `${x * 4},${y}`;
                    })
                    .join(" ")}
                  style={{ filter: "drop-shadow(0 0 4px var(--accent-glow))" }}
                />
              </svg>
            </div>
          )}
        </div>
      )}

      {/* Evaluation */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 20 }}>
        <h2 style={sectionHeadingStyle}>Evaluation</h2>
        <button className="btn-gold" onClick={startEval} disabled={evaluating} style={{ padding: "8px 20px", marginBottom: 12, fontSize: 14, opacity: evaluating ? 0.5 : 1 }}>
          {evaluating ? "⏳ Evaluating..." : "► Run Evaluation"}
        </button>
        {evalError && (
          <div style={{ padding: 10, marginBottom: 12, borderLeft: "3px solid var(--error)", color: "var(--error)", fontSize: 13, fontWeight: 600 }}>
            ✗ Evaluation failed: {evalError}
          </div>
        )}
        {evalResult && !evalError && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "8px 0", color: XBOX.dimGreen }}>Task</th>
                  <th style={{ textAlign: "right", padding: "8px 0", color: XBOX.dimGreen }}>Score</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: "8px 0" }}>{evalResult.task || "mmlu_stem"}</td>
                  <td style={{ textAlign: "right", padding: "8px 0" }}>
                    <span style={{ color: XBOX.chartreuse, fontWeight: 700 }}>{(evalResult.score * 100).toFixed(1)}%</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </motion.div>
        )}
      </div>
    </div>
  );
}
