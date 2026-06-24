import { useRef, useCallback, useState, useEffect } from "react";

type SoundType = "sweep" | "select" | "scroll" | "back";

/**
 * Web Audio API sound effect manager for Xbox Blades UI.
 * Generates all sounds programmatically — no audio files needed.
 */
export class SoundEngine {
  private ctx: AudioContext | null = null;
  private muted = false;
  private ambientNodes: { osc: OscillatorNode; gain: GainNode } | null = null;

  constructor() {
    // Load mute preference
    try {
      this.muted = localStorage.getItem("mindforge-muted") === "true";
    } catch {}
  }

  private ensureCtx(): AudioContext | null {
    if (!this.ctx) {
      try {
        this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      } catch {
        return null;
      }
    }
    if (this.ctx.state === "suspended") {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  setMuted(muted: boolean) {
    this.muted = muted;
    try {
      localStorage.setItem("mindforge-muted", String(muted));
    } catch {}
    if (muted) this.stopAmbient();
  }

  isMuted() {
    return this.muted;
  }

  /** Blade sweep: filtered noise burst with pitch sweep (whoosh) */
  sweep() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.35;

    // Create noise buffer
    const bufferSize = ctx.sampleRate * duration;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
    }

    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    // Bandpass filter for whoosh effect
    const filter = ctx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.setValueAtTime(400, now);
    filter.frequency.exponentialRampToValueAtTime(2000, now + duration);
    filter.Q.value = 0.8;

    // Gain envelope
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.15, now + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);
    noise.start(now);
    noise.stop(now + duration);
  }

  /** Menu select: short sine wave ping at 800Hz, 50ms decay */
  select() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = 800;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.05);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.05);
  }

  /** Menu scroll: very short tick at 1200Hz, 20ms */
  scroll() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = "square";
    osc.frequency.value = 1200;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.02);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.02);
  }

  /** Back: reverse whoosh (pitch sweeps down) */
  back() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.3;

    const bufferSize = ctx.sampleRate * duration;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (i / bufferSize);
    }

    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.setValueAtTime(2000, now);
    filter.frequency.exponentialRampToValueAtTime(200, now + duration);
    filter.Q.value = 0.8;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.15, now + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);
    noise.start(now);
    noise.stop(now + duration);
  }

  /** Low-volume ambient drone — can be toggled */
  startAmbient() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx || this.ambientNodes) return;

    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = 55; // low drone

    const gain = ctx.createGain();
    gain.gain.value = 0.02;

    // LFO for subtle modulation
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.1;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = 0.01;
    lfo.connect(lfoGain);
    lfoGain.connect(gain.gain);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    lfo.start();
    this.ambientNodes = { osc, gain };
  }

  stopAmbient() {
    if (this.ambientNodes) {
      try {
        this.ambientNodes.osc.stop();
      } catch {}
      this.ambientNodes = null;
    }
  }

  play(type: SoundType) {
    switch (type) {
      case "sweep": this.sweep(); break;
      case "select": this.select(); break;
      case "scroll": this.scroll(); break;
      case "back": this.back(); break;
    }
  }
}

// Singleton instance
let soundEngine: SoundEngine | null = null;

export function getSoundEngine(): SoundEngine {
  if (!soundEngine) {
    soundEngine = new SoundEngine();
  }
  return soundEngine;
}

/**
 * Mute toggle button component.
 */
export function MuteToggle() {
  const [muted, setMuted] = useState(() => getSoundEngine().isMuted());

  const toggle = () => {
    const newMuted = !muted;
    setMuted(newMuted);
    getSoundEngine().setMuted(newMuted);
    if (!newMuted) {
      getSoundEngine().play("select");
    }
  };

  return (
    <button
      onClick={toggle}
      aria-label={muted ? "Unmute sound effects" : "Mute sound effects"}
      style={{
        background: "transparent",
        border: "1px solid var(--border)",
        borderRadius: 4,
        padding: "4px 8px",
        cursor: "pointer",
        fontSize: 14,
        color: muted ? "var(--text-dim)" : "var(--accent)",
        display: "flex",
        alignItems: "center",
        gap: 4,
      }}
    >
      {muted ? "🔇" : "🔊"}
    </button>
  );
}
