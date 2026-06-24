import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { apiPost } from "../lib/api";
import { useWebSocket } from "../hooks/useWebSocket";

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
    <div style={{ padding: 24, height: "100%", overflowY: "auto" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Train & Evaluate</h1>

      {/* Training Error */}
      {trainError && (
        <div className="panel" style={{ padding: 12, marginBottom: 16, borderLeft: "3px solid var(--error)" }}>
          <span style={{ color: "var(--error)", fontSize: 14, fontWeight: 600 }}>✗ Training failed: {trainError}</span>
          <button
            onClick={() => { setTrainError(null); startTrain(); }}
            style={{ marginLeft: 12, padding: "4px 12px", fontSize: 12, background: "var(--surface-raised)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer" }}
          >
            ↻ Retry
          </button>
        </div>
      )}

      {/* Base Model Selection */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>Base Model</h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "var(--accent)" }}>●</span>
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{ flex: 1, padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }}
          />
        </div>
      </div>

      {/* Training Config */}
      <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>Training Config</h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
          <div>
            <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Method</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)} style={{ width: "100%", padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }}>
              <option value="dpo">DPO (preference)</option>
              <option value="sft">SFT (supervised)</option>
              <option value="orpo">ORPO</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Adapter</label>
            <select value={adapter} onChange={(e) => setAdapter(e.target.value)} style={{ width: "100%", padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }}>
              <option value="lora">LoRA</option>
              <option value="dora">DoRA</option>
              <option value="full">Full</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Data Path</label>
            <input value={data} onChange={(e) => setData(e.target.value)} style={{ width: "100%", padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Iterations</label>
            <input type="number" value={iters} onChange={(e) => setIters(+e.target.value)} style={{ width: "100%", padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Batch Size</label>
            <input type="number" value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} style={{ width: "100%", padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Learning Rate</label>
            <input type="number" step="0.00001" value={learningRate} onChange={(e) => setLearningRate(+e.target.value)} style={{ width: "100%", padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }} />
          </div>
          {mode === "dpo" && (
            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>Beta (DPO)</label>
              <input type="number" step="0.01" value={beta} onChange={(e) => setBeta(+e.target.value)} style={{ width: "100%", padding: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)" }} />
            </div>
          )}
        </div>
        <button className="btn-gold gold-glow" onClick={startTrain} disabled={running} style={{ width: "100%", padding: 12, fontSize: 16, opacity: running ? 0.5 : 1 }}>
          {running ? "⏸ Training..." : "► Start Fine-Tuning"}
        </button>
      </div>

      {/* Training Progress */}
      {(running || currentIter > 0) && (
        <div className="panel" style={{ padding: 20, marginBottom: 20 }}>
          <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>Training Progress</h3>
          <div style={{ fontSize: 14, marginBottom: 8 }}>
            Iteration <span style={{ color: "var(--accent)" }}>{currentIter}</span> / {iters}
          </div>
          <div className="progress-bar" style={{ height: 12, marginBottom: 8 }}>
            <motion.div className="progress-fill" animate={{ width: `${pct}%` }} style={{ width: `${pct}%` }} />
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 12 }}>
            {pct.toFixed(1)}% · Loss: <span style={{ color: "var(--accent)" }}>{currentLoss.toFixed(4)}</span>
            {lossDelta !== 0 && (
              <span style={{ color: lossDelta < 0 ? "var(--success)" : "var(--error)", marginLeft: 8 }}>
                ({lossDelta < 0 ? "↓" : "↑"} {Math.abs(lossDelta).toFixed(4)})
              </span>
            )}
          </div>

          {/* Loss Curve */}
          {lossHistory.length > 1 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>Loss Curve</div>
              <svg width="100%" height="120" style={{ background: "var(--surface-raised)", borderRadius: 4 }}>
                <polyline
                  fill="none"
                  stroke="var(--accent)"
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
      <div className="panel" style={{ padding: 20 }}>
        <h3 style={{ marginBottom: 12, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>Evaluation</h3>
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
                  <th style={{ textAlign: "left", padding: "8px 0", color: "var(--text-secondary)" }}>Task</th>
                  <th style={{ textAlign: "right", padding: "8px 0", color: "var(--text-secondary)" }}>Score</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: "8px 0" }}>{evalResult.task || "mmlu_stem"}</td>
                  <td style={{ textAlign: "right", padding: "8px 0" }}>
                    <span style={{ color: "var(--accent)", fontWeight: 700 }}>{(evalResult.score * 100).toFixed(1)}%</span>
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
