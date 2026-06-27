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

// Framer Motion variants for blade sweep transition (3D perspective)
const bladeVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? "100%" : "-100%",
    rotateY: direction > 0 ? 20 : -20,
    scale: 1,
    filter: "blur(0px)",
    opacity: 0,
  }),
  center: {
    x: 0,
    rotateY: 0,
    scale: 1,
    filter: "blur(0px)",
    opacity: 1,
  },
  exit: (direction: number) => ({
    x: direction > 0 ? "-30%" : "30%",
    rotateY: direction > 0 ? -20 : 20,
    scale: 0.92,
    filter: "blur(4px)",
    opacity: 0,
  }),
};

export default function App() {
  const [screen, setScreen] = useState<Screen>("model-setup");
  const [connected, setConnected] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [direction, setDirection] = useState(0);
  const [showBoot, setShowBoot] = useState(true);
  const prefersReducedMotion = useReducedMotion();

  // Boot sequence: play boot sound, then dismiss after 1.5s
  useEffect(() => {
    getSoundEngine().play("boot");
    const timer = setTimeout(() => setShowBoot(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  // Start ambient drone after boot screen dismisses
  useEffect(() => {
    if (!showBoot) {
      getSoundEngine().startAmbient();
    }
  }, [showBoot]);

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
    getSoundEngine().play("whoosh");
    setScreen(target);
  }, [screen]);

  // Arrow key navigation between blades (up/down for vertical menu)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't interfere with typing
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      // Don't interfere with modifier keys
      if (e.ctrlKey || e.metaKey) return;
      // Review Dashboard uses ArrowLeft/Right for item navigation
      if (screen === "review") return;

      const currentIdx = SCREEN_ORDER.indexOf(screen);
      if (e.key === "ArrowDown") {
        if (currentIdx < SCREEN_ORDER.length - 1) {
          e.preventDefault();
          navigate(SCREEN_ORDER[currentIdx + 1]);
        }
      } else if (e.key === "ArrowUp") {
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
            background: "linear-gradient(180deg, rgba(4,28,28,0.95) 0%, transparent 100%)",
            position: "relative",
            zIndex: 30,
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="caduceus" style={{ fontSize: 20 }}>⚕</span>
            <span style={{ fontSize: 16, fontWeight: 700, color: "var(--accent)" }}>MindForge</span>
            <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 8 }}>v0.0.1</span>
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

        {/* Main area: content (left) + vertical blade bar (right) */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative" }}>
          {/* Left: blade content area with 3D perspective */}
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
            {/* Wireframe perspective grid background */}
            <div className="wireframe-grid" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0 }} />

            {/* Hexagonal grid background overlay */}
            <div className="hex-grid" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 1 }} />

            {/* Xbox orb — large pulsing sphere in background */}
            <div
              className="xbox-orb orb-pulse"
              style={{
                position: "absolute",
                top: "50%",
                left: "35%",
                transform: "translate(-50%, -50%)",
                pointerEvents: "none",
                zIndex: 0,
              }}
              aria-hidden="true"
            />

            {/* Scanline overlay */}
            <div className="scanlines" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 2 }} />

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
                transition={prefersReducedMotion ? { duration: 0.15 } : { duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                style={{
                  height: "100%",
                  transformStyle: "preserve-3d",
                  backfaceVisibility: "hidden",
                  position: "relative",
                  zIndex: 3,
                  willChange: "transform, opacity, filter",
                }}
              >
                <main id="main-content" role="region" aria-label={SCREEN_TITLES[screen]} aria-live="polite" style={{ height: "100%" }}>
                  {screens[screen]}
                </main>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Right: vertical blade bar */}
          <div
            style={{
              width: 240,
              flexShrink: 0,
              position: "relative",
              zIndex: 20,
            }}
          >
            <BladeBar active={screen} onSelect={navigate} direction={direction} />
          </div>
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
          B = Back
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
          A = Select
        </div>

        {/* Boot screen overlay */}
        {showBoot && (
          <div
            className="boot-screen"
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 100,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div className="hex-grid" style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />
            <div className="scanlines" style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />
            <div
              className="xbox-orb orb-pulse"
              style={{ position: "absolute", pointerEvents: "none" }}
              aria-hidden="true"
            />
            <span className="caduceus" style={{ fontSize: 48, position: "relative", zIndex: 1 }}>⚕</span>
            <div
              style={{
                fontSize: 24,
                fontWeight: 700,
                color: "var(--accent)",
                marginTop: 16,
                textShadow: "0 0 20px var(--accent-glow)",
                position: "relative",
                zIndex: 1,
              }}
            >
              MindForge
            </div>
            <div
              style={{
                fontSize: 12,
                color: "var(--text-dim)",
                marginTop: 8,
                position: "relative",
                zIndex: 1,
              }}
            >
              Initializing MindForge...
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}
