// Hermes-themed color palette for MindForge
// Based on the design doc: deep dark teal background, warm gold accents

export const theme = {
  background: "#041C1C",        // Hermes "Swamp" — deep dark teal
  surface: "#1B1713",             // Hermes "amber-800" — dark warm brown
  surfaceRaised: "#363029",       // Hermes "amber-700" — elevated cards
  accent: "#FFD700",              // Hermes gold — primary
  accentGlow: "#FFD70044",        // gold with 27% opacity for glows
  accentDim: "#CD7F32",           // Hermes "banner_border" — bronze
  accentSecondary: "#C4DA7D",     // Hermes "lime-100" — yellow-green
  textPrimary: "#FFF8DC",         // Hermes "banner_text" — cornsilk white
  textSecondary: "#B8860B",       // Hermes "banner_dim" — dark goldenrod
  textDim: "#918270",             // Lightened for WCAG AA (was #544B41, 2.07:1)
  success: "#C4DA7D",             // Hermes lime-100 — green-ish for correct
  warning: "#FFBF00",             // Hermes "banner_accent" — amber
  error: "#D26464",               // Lightened for WCAG AA (was #CD5C5C, 4.45:1)
  info: "#4DD0E1",                // Hermes "ui_label" — cyan
  border: "#CD7F32",              // Hermes bronze — panel borders
} as const;

// Theme variants (selectable in Settings)
export const themeVariants: Record<string, Record<string, string>> = {
  gold: theme,
  cyberpunk: {
    ...theme,
    background: "#0a0a0f",
    surface: "#11111b",
    surfaceRaised: "#1a1a2e",
    accent: "#ff00ff",
    accentGlow: "#ff00ff44",
    accentDim: "#7f007f",
    textPrimary: "#e0e0ff",
    border: "#5555ff",
  },
  slate: {
    ...theme,
    background: "#0d1117",
    surface: "#161b22",
    surfaceRaised: "#21262d",
    accent: "#4169E1",
    accentGlow: "#4169E144",
    accentDim: "#2b4a8f",
    border: "#3b5998",
    textPrimary: "#c9d1d9",
  },
  mono: {
    ...theme,
    background: "#1a1a1a",
    surface: "#222222",
    surfaceRaised: "#333333",
    accent: "#cccccc",
    accentGlow: "#cccccc44",
    accentDim: "#888888",
    border: "#555555",
    textPrimary: "#c9d1d9",
    textSecondary: "#999999",
  },
};

export type Screen =
  | "model-setup"
  | "domain-setup"
  | "probing"
  | "review"
  | "format"
  | "train"
  | "stats"
  | "settings";
