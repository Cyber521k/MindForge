import { useState, useMemo, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { Screen } from "./lib/theme";
import { Sidebar } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
import { ErrorBoundary } from "./components/ErrorBoundary";
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

// Framer Motion variants for blade sweep transition
const bladeVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? "100%" : "-100%",
    rotateY: direction > 0 ? 45 : -45,
    opacity: 0,
  }),
  center: {
    x: 0,
    rotateY: 0,
    opacity: 1,
  },
  exit: (direction: number) => ({
    x: direction > 0 ? "-100%" : "100%",
    rotateY: direction > 0 ? -45 : 45,
    opacity: 0,
  }),
};

export default function App() {
  const [screen, setScreen] = useState<Screen>("model-setup");
  const [connected, setConnected] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [direction, setDirection] = useState(0);

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

  // Navigate with direction awareness for blade sweep
  const navigate = useCallback((target: Screen) => {
    const currentIdx = SCREEN_ORDER.indexOf(screen);
    const targetIdx = SCREEN_ORDER.indexOf(target);
    setDirection(targetIdx > currentIdx ? 1 : -1);
    setScreen(target);
  }, [screen]);

  // Arrow key navigation between blades (left/right only — up/down reserved for in-screen nav)
  // Skip on review screen — it uses arrows for item navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't interfere with typing
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      // Don't interfere with modifier keys (Ctrl+S, etc.)
      if (e.ctrlKey || e.metaKey) return;
      // Review Dashboard uses ArrowLeft/Right/Up/Down for item navigation
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

  // Memoize so screen elements are stable references
  const screens = useMemo<Record<Screen, JSX.Element>>(() => ({
    "model-setup": <ModelSetup onSelectModel={setSelectedModel} />,
    "domain-setup": <DomainSetup />,
    "probing": <ProbingProgress />,
    "review": <ReviewDashboard />,
    "format": <FormatExport />,
    "train": <TrainEvaluate />,
    "stats": <Stats />,
    "settings": <Settings />,
  }), []);

  return (
    <ErrorBoundary>
      <div style={{
        display: "flex",
        height: "100vh",
        background: "var(--bg)",
        perspective: "1400px",
        perspectiveOrigin: "center center",
      }}>
        <Sidebar active={screen} onSelect={navigate} model={selectedModel} />
        <div style={{ flex: 1, display: "flex", flexDirection: "column", transformStyle: "preserve-3d" }}>
          <main
            id="main-content"
            role="region"
            aria-label="Screen content"
            aria-live="polite"
            style={{
              flex: 1,
              position: "relative",
              overflow: "hidden",
              transformStyle: "preserve-3d",
            }}
          >
            <AnimatePresence mode="wait" custom={direction}>
              <motion.div
                key={screen}
                custom={direction}
                variants={bladeVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={{
                  x: { type: "spring", stiffness: 300, damping: 30 },
                  rotateY: { type: "spring", stiffness: 300, damping: 30 },
                  opacity: { duration: 0.2 },
                }}
                style={{
                  height: "100%",
                  transformStyle: "preserve-3d",
                  backfaceVisibility: "hidden",
                }}
              >
                {screens[screen]}
              </motion.div>
            </AnimatePresence>
          </main>
          <StatusBar phase={screen} model={selectedModel} connected={connected} />
        </div>
      </div>
    </ErrorBoundary>
  );
}
