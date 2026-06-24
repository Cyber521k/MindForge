import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { Screen } from "./lib/theme";
import { Sidebar } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
import { ModelSetup } from "./screens/ModelSetup";
import { DomainSetup } from "./screens/DomainSetup";
import { ProbingProgress } from "./screens/ProbingProgress";
import { ReviewDashboard } from "./screens/ReviewDashboard";
import { FormatExport } from "./screens/FormatExport";
import { TrainEvaluate } from "./screens/TrainEvaluate";
import { Stats } from "./screens/Stats";
import { Settings } from "./screens/Settings";

export default function App() {
  const [screen, setScreen] = useState<Screen>("model-setup");

  const screens: Record<Screen, JSX.Element> = {
    "model-setup": <ModelSetup />,
    "domain-setup": <DomainSetup />,
    "probing": <ProbingProgress />,
    "review": <ReviewDashboard />,
    "format": <FormatExport />,
    "train": <TrainEvaluate />,
    "stats": <Stats />,
    "settings": <Settings />,
  };

  return (
    <div style={{ display: "flex", height: "100vh", background: "var(--bg)" }}>
      <Sidebar active={screen} onSelect={setScreen} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
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
        </div>
        <StatusBar phase={screen} />
      </div>
    </div>
  );
}
