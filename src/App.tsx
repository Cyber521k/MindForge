import { useState, useMemo, useEffect } from "react";
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

export default function App() {
  const [screen, setScreen] = useState<Screen>("model-setup");
  const [connected, setConnected] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("");

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

  // Memoize so screen elements are stable references (avoids re-creating all screens on every render)
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
    <div style={{ display: "flex", height: "100vh", background: "var(--bg)" }}>
      <Sidebar active={screen} onSelect={setScreen} model={selectedModel} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <main id="main-content" style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={screen}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              style={{ height: "100%" }}
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
