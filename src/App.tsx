import { useState, useMemo, useEffect, useCallback, useRef } from "react";
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

const XBOX_NEON_GREEN = "var(--xbox-neon-green, #35ff4d)";
const XBOX_NEON_GREEN_DIM = "rgba(53, 255, 77, 0.28)";
const BOOT_SPINNER_DARK = "rgba(53, 255, 77, 0.18)";

type BootPhase = "logo" | "spinner" | "dashboard";

// Xbox 2001-style fade-through-black blade transition.
const bladeVariants = {
  enter: {
    opacity: 0,
    rotateY: 0,
    transition: { duration: 0.2, delay: 0.1, ease: [0.22, 1, 0.36, 1] },
  },
  center: {
    opacity: 1,
    rotateY: 0,
    transition: { duration: 0.2, delay: 0.1, ease: [0.22, 1, 0.36, 1] },
  },
  exit: (direction: number) => ({
    opacity: 0,
    rotateY: direction >= 0 ? -5 : 5,
    transition: { duration: 0.15, ease: "easeOut" },
  }),
};

const reducedBladeVariants = {
  enter: { opacity: 1, rotateY: 0, transition: { duration: 0 } },
  center: { opacity: 1, rotateY: 0, transition: { duration: 0 } },
  exit: { opacity: 1, rotateY: 0, transition: { duration: 0 } },
};

function XboxBootLogo() {
  return (
    <div
      aria-label="Xbox boot logo"
      role="img"
      style={{
        width: 132,
        height: 132,
        position: "relative",
        filter: "drop-shadow(0 0 28px rgba(53, 255, 77, 0.65))",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: "14px 56px",
          borderRadius: 999,
          background: `linear-gradient(180deg, transparent 0%, ${XBOX_NEON_GREEN} 18%, ${XBOX_NEON_GREEN} 82%, transparent 100%)`,
          transform: "rotate(42deg)",
          transformOrigin: "center",
        }}
      />
      <span
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: "14px 56px",
          borderRadius: 999,
          background: `linear-gradient(180deg, transparent 0%, ${XBOX_NEON_GREEN} 18%, ${XBOX_NEON_GREEN} 82%, transparent 100%)`,
          transform: "rotate(-42deg)",
          transformOrigin: "center",
        }}
      />
    </div>
  );
}

function SegmentedBootSpinner({ reducedMotion }: { reducedMotion: boolean }) {
  return (
    <motion.div
      aria-label="Loading"
      role="status"
      animate={reducedMotion ? undefined : { rotate: 360 }}
      transition={reducedMotion ? undefined : { duration: 0.9, repeat: Infinity, ease: "linear" }}
      style={{
        width: 122,
        height: 122,
        borderRadius: "50%",
        background: `conic-gradient(${XBOX_NEON_GREEN} 0deg 60deg, ${BOOT_SPINNER_DARK} 60deg 120deg, ${XBOX_NEON_GREEN} 120deg 180deg, ${BOOT_SPINNER_DARK} 180deg 240deg, ${XBOX_NEON_GREEN} 240deg 300deg, ${BOOT_SPINNER_DARK} 300deg 360deg)`,
        WebkitMask: "radial-gradient(circle, transparent 0 44%, #000 45% 64%, transparent 65%)",
        mask: "radial-gradient(circle, transparent 0 44%, #000 45% 64%, transparent 65%)",
        boxShadow: "0 0 38px rgba(53, 255, 77, 0.45)",
      }}
    />
  );
}

function XboxFooter() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        height: 54,
        zIndex: 25,
        pointerEvents: "none",
        color: XBOX_NEON_GREEN,
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
        letterSpacing: 0,
        textShadow: "0 0 12px rgba(53, 255, 77, 0.65)",
      }}
    >
      <svg
        viewBox="0 0 1000 54"
        preserveAspectRatio="none"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      >
        <path
          className="xbox-circuit-trace"
          d="M72 38 C190 38 230 22 350 22 L478 22"
          fill="none"
          stroke={XBOX_NEON_GREEN}
          strokeWidth="1"
          strokeLinecap="round"
          opacity="0.7"
        />
        <path
          className="xbox-circuit-trace"
          d="M928 38 C810 38 770 22 650 22 L522 22"
          fill="none"
          stroke={XBOX_NEON_GREEN}
          strokeWidth="1"
          strokeLinecap="round"
          opacity="0.7"
        />
      </svg>
      <div
        style={{
          position: "absolute",
          left: 20,
          bottom: 12,
          display: "flex",
          alignItems: "center",
          gap: 7,
        }}
      >
        <span
          style={{
            width: 18,
            height: 18,
            borderRadius: "50%",
            border: `1px solid ${XBOX_NEON_GREEN}`,
            display: "grid",
            placeItems: "center",
            boxShadow: "0 0 12px rgba(53, 255, 77, 0.35)",
          }}
        >
          B
        </span>
        <span>B - BACK</span>
      </div>
      <div
        style={{
          position: "absolute",
          right: 20,
          bottom: 12,
          display: "flex",
          alignItems: "center",
          gap: 7,
        }}
      >
        <span>A - SELECT</span>
        <span
          style={{
            width: 18,
            height: 18,
            borderRadius: "50%",
            border: `1px solid ${XBOX_NEON_GREEN}`,
            display: "grid",
            placeItems: "center",
            boxShadow: "0 0 12px rgba(53, 255, 77, 0.35)",
          }}
        >
          A
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [screen, setScreen] = useState<Screen>("model-setup");
  const [connected, setConnected] = useState(false);
  const [controllerConnected, setControllerConnected] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [direction, setDirection] = useState(0);
  const [showBoot, setShowBoot] = useState(true);
  const [bootPhase, setBootPhase] = useState<BootPhase>("logo");
  const [transitionBlack, setTransitionBlack] = useState(false);
  const previousGamepadButtons = useRef<Set<number>>(new Set());
  const transitionTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prefersReducedMotion = useReducedMotion();
  const reduceMotion = Boolean(prefersReducedMotion);

  const particles = useMemo(() => (
    Array.from({ length: 7 }, (_, index) => ({
      id: index,
      left: 10 + Math.random() * 78,
      top: 14 + Math.random() * 68,
      size: 2 + Math.random() * 3,
      driftX: -10 + Math.random() * 20,
      driftY: -14 - Math.random() * 18,
      duration: 5.5 + Math.random() * 4,
      delay: Math.random() * 2.5,
    }))
  ), []);

  // Boot sequence: Xbox-style logo, spinner, then dashboard reveal.
  useEffect(() => {
    getSoundEngine().play("boot");
    const spinnerTimer = setTimeout(() => setBootPhase("spinner"), 500);
    const dashboardTimer = setTimeout(() => {
      setBootPhase("dashboard");
      getSoundEngine().startAmbient();
    }, 1500);
    const dismissTimer = setTimeout(() => setShowBoot(false), 2000);
    return () => {
      clearTimeout(spinnerTimer);
      clearTimeout(dashboardTimer);
      clearTimeout(dismissTimer);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (transitionTimer.current) {
        clearTimeout(transitionTimer.current);
      }
    };
  }, []);

  const dispatchKeyboard = useCallback((key: string) => {
    const event = new KeyboardEvent("keydown", {
      key,
      bubbles: true,
      cancelable: true,
    });
    const target = document.activeElement;
    if (target && target !== document.body && target !== document.documentElement) {
      target.dispatchEvent(event);
      return;
    }
    window.dispatchEvent(event);
  }, []);

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
    if (target === screen) return;

    const currentIdx = SCREEN_ORDER.indexOf(screen);
    const targetIdx = SCREEN_ORDER.indexOf(target);
    const dir = targetIdx > currentIdx ? 1 : -1;
    setDirection(dir);

    if (!reduceMotion) {
      setTransitionBlack(true);
      if (transitionTimer.current) {
        clearTimeout(transitionTimer.current);
      }
      transitionTimer.current = setTimeout(() => setTransitionBlack(false), 480);
    } else {
      setTransitionBlack(false);
    }

    getSoundEngine().play("sweep");
    getSoundEngine().play("whoosh");
    setScreen(target);
  }, [reduceMotion, screen]);

  // Gamepad API support: edge-triggered standard-controller buttons.
  useEffect(() => {
    if (typeof navigator === "undefined" || typeof navigator.getGamepads !== "function") return;

    let frameId = 0;
    const pollGamepads = () => {
      const gamepads = Array.from(navigator.getGamepads()).filter(Boolean) as Gamepad[];
      const gamepad = gamepads[0];
      const isConnected = Boolean(gamepad);
      setControllerConnected((current) => (current === isConnected ? current : isConnected));

      if (!gamepad) {
        previousGamepadButtons.current.clear();
        frameId = window.requestAnimationFrame(pollGamepads);
        return;
      }

      const pressed = new Set<number>();
      gamepad.buttons.forEach((button, index) => {
        if (button.pressed) pressed.add(index);
      });

      const justPressed = (index: number) => (
        pressed.has(index) && !previousGamepadButtons.current.has(index)
      );
      const currentIdx = SCREEN_ORDER.indexOf(screen);

      if (justPressed(14) && currentIdx > 0) {
        navigate(SCREEN_ORDER[currentIdx - 1]);
      }
      if (justPressed(15) && currentIdx < SCREEN_ORDER.length - 1) {
        navigate(SCREEN_ORDER[currentIdx + 1]);
      }
      if (justPressed(12)) dispatchKeyboard("ArrowUp");
      if (justPressed(13)) dispatchKeyboard("ArrowDown");
      if (justPressed(0)) dispatchKeyboard("Enter");
      if (justPressed(1)) dispatchKeyboard("Escape");

      previousGamepadButtons.current = pressed;
      frameId = window.requestAnimationFrame(pollGamepads);
    };

    frameId = window.requestAnimationFrame(pollGamepads);
    return () => window.cancelAnimationFrame(frameId);
  }, [dispatchKeyboard, navigate, screen]);

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

  const dashboardVisible = !showBoot || bootPhase === "dashboard";
  const dashboardMotion = (delay: number, y: number) => ({
    opacity: dashboardVisible ? 1 : 0,
    y: dashboardVisible ? 0 : y,
    transition: { duration: reduceMotion ? 0 : 0.34, delay: dashboardVisible && !reduceMotion ? delay : 0, ease: "easeOut" },
  });

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
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: -8 }}
          animate={dashboardMotion(0, -8)}
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
            {controllerConnected && (
              <span style={{ fontSize: 11, color: XBOX_NEON_GREEN }}>
                ● Controller
              </span>
            )}
            <span style={{ fontSize: 11, color: connected ? "var(--success)" : "var(--error)" }}>
              {connected ? "● Connected" : "○ Disconnected"}
            </span>
            <MuteToggle />
          </div>
        </motion.div>

        {/* Main area: content (left) + vertical blade bar (right) */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative" }}>
          {/* Left: blade content area with 3D perspective */}
          <motion.div
            initial={reduceMotion ? false : { opacity: 0, y: 12 }}
            animate={dashboardMotion(0.08, 12)}
            style={{
              flex: 1,
              position: "relative",
              overflow: "hidden",
              perspective: "1200px",
              perspectiveOrigin: "center 40%",
              transformStyle: "preserve-3d",
            }}
          >
            {/* Xbox perspective floor background */}
            <div
              className="xbox-perspective-floor"
              style={{
                position: "absolute",
                left: "-12%",
                right: "-12%",
                bottom: "-18%",
                height: "48%",
                transform: "perspective(720px) rotateX(68deg)",
                transformOrigin: "center bottom",
                background: `repeating-linear-gradient(90deg, transparent 0 42px, ${XBOX_NEON_GREEN_DIM} 42px 43px), repeating-linear-gradient(0deg, transparent 0 34px, ${XBOX_NEON_GREEN_DIM} 34px 35px)`,
                opacity: 0.32,
                filter: "drop-shadow(0 0 16px rgba(53, 255, 77, 0.18))",
                pointerEvents: "none",
                zIndex: 0,
              }}
            />

            {/* Wireframe perspective grid background */}
            <div className="wireframe-grid" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0 }} />

            {/* Hexagonal grid background overlay */}
            <div className="hex-grid" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 1 }} />

            {/* Floating Xbox particles */}
            {particles.map((particle) => (
              <motion.div
                key={particle.id}
                className="xbox-particle-float"
                aria-hidden="true"
                animate={reduceMotion ? { opacity: 0.35 } : {
                  opacity: [0.14, 0.65, 0.14],
                  x: [0, particle.driftX, 0],
                  y: [0, particle.driftY, 0],
                }}
                transition={reduceMotion ? { duration: 0 } : {
                  duration: particle.duration,
                  delay: particle.delay,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                style={{
                  position: "absolute",
                  left: `${particle.left}%`,
                  top: `${particle.top}%`,
                  width: particle.size,
                  height: particle.size,
                  borderRadius: "50%",
                  background: XBOX_NEON_GREEN,
                  boxShadow: "0 0 12px rgba(53, 255, 77, 0.75)",
                  pointerEvents: "none",
                  zIndex: 2,
                }}
              />
            ))}

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

            {transitionBlack && (
              <div
                aria-hidden="true"
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "#000",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
            )}

            <AnimatePresence mode="wait" custom={direction}>
              <motion.div
                key={screen}
                custom={direction}
                variants={reduceMotion ? reducedBladeVariants : bladeVariants}
                initial="enter"
                animate="center"
                exit="exit"
                style={{
                  height: "100%",
                  transformStyle: "preserve-3d",
                  backfaceVisibility: "hidden",
                  position: "relative",
                  zIndex: 4,
                  willChange: "transform, opacity",
                }}
              >
                <main id="main-content" role="region" aria-label={SCREEN_TITLES[screen]} aria-live="polite" style={{ height: "100%" }}>
                  {screens[screen]}
                </main>
              </motion.div>
            </AnimatePresence>
          </motion.div>

          {/* Right: vertical blade bar */}
          <motion.div
            initial={reduceMotion ? false : { opacity: 0, y: 16 }}
            animate={dashboardMotion(0.18, 16)}
            style={{
              width: 240,
              flexShrink: 0,
              position: "relative",
              zIndex: 20,
            }}
          >
            <BladeBar active={screen} onSelect={navigate} direction={direction} />
          </motion.div>
        </div>

        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: 10 }}
          animate={dashboardMotion(0.25, 10)}
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 25,
            pointerEvents: "none",
          }}
        >
          <XboxFooter />
        </motion.div>

        {/* Boot screen overlay */}
        {showBoot && (
          <motion.div
            className="boot-screen"
            initial={{ opacity: 1 }}
            animate={{ opacity: bootPhase === "dashboard" ? 0 : 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.5, ease: "easeOut" }}
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 100,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "#000",
            }}
          >
            <AnimatePresence mode="wait">
              {bootPhase === "logo" && (
                <motion.div
                  key="boot-logo"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: reduceMotion ? 0 : 0.18 }}
                >
                  <XboxBootLogo />
                </motion.div>
              )}
              {bootPhase === "spinner" && (
                <motion.div
                  key="boot-spinner"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: reduceMotion ? 0 : 0.18 }}
                >
                  <SegmentedBootSpinner reducedMotion={reduceMotion} />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </div>
    </ErrorBoundary>
  );
}
