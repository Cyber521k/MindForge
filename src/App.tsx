import { useState, useMemo, useEffect, useCallback } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { Screen } from "./lib/theme";
import { BladeBar } from "./components/BladeBar";
import { BladeContent } from "./components/BladeContent";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { MuteToggle, getSoundEngine } from "./components/SoundManager";
import { ModelSetup } from "./screens/ModelSetup";
import { DomainSetup } from "./screens/DomainSetup";
import { ProbingProgress } from "./screens/ProbingProgress";
import { ReviewDashboard } from "./screens/ReviewDashboard";
import { FormatExport } from "./screens/FormatExport";
import { TrainEvaluate } from "./screens/TrainEvaluate";
import { Stats } from "./screens/Stats";
import { Settings } from "./screens/Settings";
import { apiGet } from "./lib/api";

const SCREEN_ORDER: Screen[] = [
  "model-setup", "domain-setup", "probing", "review",
  "format", "train", "stats", "settings",
];

const SCREEN_ICONS: Record<Screen, string> = {
  "model-setup": "🖥",
  "domain-setup": "📚",
  "probing": "🔍",
  "review": "📋",
  "format": "📦",
  "train": "🎯",
  "stats": "📊",
  "settings": "⚙",
};

const SCREEN_TITLES: Record<Screen, string> = {
  "model-setup": "Model Setup",
  "domain-setup": "Domain Setup",
  "probing": "Probe Engine",
  "review": "Review Dashboard",
  "format": "Format & Export",
  "train": "Train & Evaluate",
  "stats": "Statistics",
  "settings": "Settings",
};

// Framer Motion variants for blade sweep transition (horizontal)
const bladeVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? "100%" : "-100%",
    rotateY: direction > 0 ? 15 : -15,
    opacity: 0,
  }),
  center: {
    x: 0,
    rotateY: 0,
    opacity: 1,
  },
  exit: (direction: number) => ({
    x: direction > 0 ? "-30%" : "30%",
    rotateY: direction > 0 ? -15 : 15,
    opacity: 0,
  }),
};

export default function App() {
  const [screen, setScreen] = useState<Screen>("model-setup");
  const [connected, setConnected] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [direction, setDirection] = useState(0);
  const prefersReducedMotion = useReducedMotion();

  // Poll backend connectivity every 5s for StatusBar
  useEffect(() => {
    const check = () => {
      apiGet("/api/hardware")
        .then(() => setConnected(true))
        .catch(() => setConnected(false));
    };
    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  // Navigate with direction awareness and sound
  const navigate = useCallback((target: Screen) => {
    const currentIdx = SCREEN_ORDER.indexOf(screen);
    const targetIdx = SCREEN_ORDER.indexOf(target);
    const dir = targetIdx > currentIdx ? 1 : -1;
    setDirection(dir);
    getSoundEngine().play("sweep");
    setScreen(target);
  }, [screen]);

  // Arrow key navigation between blades (left/right only)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't interfere with typing
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      // Don't interfere with modifier keys
      if (e.ctrlKey || e.metaKey) return;
      // Review Dashboard uses ArrowLeft/Right for item navigation
      if (screen === "review") return;

      const currentIdx = SCREEN_ORDER.indexOf(screen);
      if (e.key === "ArrowRight") {
        if (currentIdx < SCREEN_ORDER.length - 1) {
          e.preventDefault();
          navigate(SCREEN_ORDER[currentIdx + 1]);
        }
      } else if (e.key === "ArrowLeft") {
        if (currentIdx > 0) {
          e.preventDefault();
          navigate(SCREEN_ORDER[currentIdx - 1]);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [screen, navigate]);

  // Memoize screen elements. Navigation callbacks are wired so "Continue"
  // buttons advance to the next blade. navigate is in deps because it
  // depends on [screen] for sweep direction.
  const screens = useMemo<Record<Screen, JSX.Element>>(() => ({
    "model-setup": <BladeContent icon="🖥" title="Model Setup"><ModelSetup onSelectModel={setSelectedModel} onContinue={() => navigate("domain-setup")} /></BladeContent>,
    "domain-setup": <BladeContent icon="📚" title="Domain Setup"><DomainSetup onStart={() => navigate("probing")} /></BladeContent>,
    "probing": <BladeContent icon="🔍" title="Probe Engine"><ProbingProgress onReview={() => navigate("review")} /></BladeContent>,
    "review": <BladeContent icon="📋" title="Review Dashboard"><ReviewDashboard onFormat={() => navigate("format")} /></BladeContent>,
    "format": <BladeContent icon="📦" title="Format & Export"><FormatExport onTrain={() => navigate("train")} /></BladeContent>,
    "train": <BladeContent icon="🎯" title="Train & Evaluate"><TrainEvaluate /></BladeContent>,
    "stats": <BladeContent icon="📊" title="Statistics"><Stats /></BladeContent>,
    "settings": <BladeContent icon="⚙" title="Settings"><Settings /></BladeContent>,
  }), [navigate]);

  return (
    <ErrorBoundary>
      <div
        className="xbox-root"
        style={{
          display: "flex",
          flexDirection: "column",
          height: "100vh",
          background: "var(--bg)",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {/* Top bar: logo + mute toggle */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "8px 20px",
            background: "linear-gradient(180deg, rgba(27,23,19,0.8) 0%, transparent 100%)",
            position: "relative",
            zIndex: 30,
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="caduceus" style={{ fontSize: 20 }}>⚕</span>
            <span style={{ fontSize: 16, fontWeight: 700, color: "var(--accent)" }}>MindForge</span>
            <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 8 }}>v7.0</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {selectedModel && (
              <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                ⚕ {selectedModel.split("/").pop()}
              </span>
            )}
            <span style={{ fontSize: 11, color: connected ? "var(--success)" : "var(--error)" }}>
              {connected ? "● Connected" : "○ Disconnected"}
            </span>
            <MuteToggle />
          </div>
        </div>

        {/* Upper 2/3: blade content area with 3D perspective */}
        <div
          style={{
            flex: 1,
            position: "relative",
            overflow: "hidden",
            perspective: "1200px",
            perspectiveOrigin: "center 40%",
            transformStyle: "preserve-3d",
          }}
        >
          {/* Hexagonal grid background overlay */}
          <div className="hex-grid" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0 }} />

          {/* Scanline overlay */}
          <div className="scanlines" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 1 }} />

          <AnimatePresence mode="wait" custom={direction}>
            <motion.div
              key={screen}
              custom={direction}
              variants={prefersReducedMotion ? {
                enter: { opacity: 0 },
                center: { opacity: 1 },
                exit: { opacity: 0 },
              } : bladeVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={prefersReducedMotion ? { duration: 0.15 } : {
                x: { type: "spring", stiffness: 260, damping: 28 },
                rotateY: { type: "spring", stiffness: 260, damping: 28 },
                opacity: { duration: 0.25 },
              }}
              style={{
                height: "100%",
                transformStyle: "preserve-3d",
                backfaceVisibility: "hidden",
                position: "relative",
                zIndex: 2,
                willChange: "transform, opacity",
              }}
            >
              <main id="main-content" role="region" aria-label={SCREEN_TITLES[screen]} aria-live="polite" style={{ height: "100%" }}>
                {screens[screen]}
              </main>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Bottom: blade bar */}
        <div
          style={{
            height: 100,
            flexShrink: 0,
            position: "relative",
            zIndex: 20,
          }}
        >
          <BladeBar active={screen} onSelect={navigate} direction={direction} />
        </div>

        {/* Controller hints */}
        <div
          style={{
            position: "absolute",
            bottom: 4,
            left: 12,
            fontSize: 10,
            color: "var(--text-dim)",
            zIndex: 25,
            pointerEvents: "none",
          }}
        >
          ← → Navigate
        </div>
        <div
          style={{
            position: "absolute",
            bottom: 4,
            right: 12,
            fontSize: 10,
            color: "var(--text-dim)",
            zIndex: 25,
            pointerEvents: "none",
          }}
        >
          Enter = Select
        </div>
      </div>
    </ErrorBoundary>
  );
}
