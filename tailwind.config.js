/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Hermes theme palette
        hermes: {
          bg: "#041C1C",
          surface: "#1B1713",
          "surface-raised": "#363029",
          accent: "#FFD700",
          "accent-glow": "#FFD70044",
          "accent-dim": "#CD7F32",
          "accent-secondary": "#C4DA7D",
          "text-primary": "#FFF8DC",
          "text-secondary": "#B8860B",
          success: "#C4DA7D",
          warning: "#FFBF00",
          error: "#D26464",
          info: "#4DD0E1",
          border: "#CD7F32",
        },
      },
      animation: {
        "pulse-gold": "pulse-gold 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow-shift": "glow-shift 3s ease-in-out infinite alternate",
        "slide-in": "slide-in 200ms ease-out",
      },
      keyframes: {
        "pulse-gold": {
          "0%, 100%": { opacity: "1", boxShadow: "0 0 8px #FFD70044" },
          "50%": { opacity: "0.7", boxShadow: "0 0 20px #FFD70088" },
        },
        "glow-shift": {
          "0%": { textShadow: "0 0 10px #FFD70044, 0 0 20px #FFD70022" },
          "100%": { textShadow: "0 0 15px #FFD70066, 0 0 30px #FFD70033" },
        },
        "slide-in": {
          "0%": { transform: "translateX(20px)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};
