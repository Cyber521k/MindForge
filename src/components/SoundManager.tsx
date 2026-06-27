import { useRef, useCallback, useState, useEffect } from "react";

type SoundType = "sweep" | "select" | "scroll" | "back" | "boot" | "whoosh";

/**
 * Web Audio API sound effect manager for Xbox Blades UI.
 * Generates all sounds programmatically — no audio files needed.
 */
export class SoundEngine {
  private ctx: AudioContext | null = null;
  private muted = false;
  private ambientNodes: { osc: OscillatorNode; lfo: OscillatorNode; gain: GainNode } | null = null;

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

  // Cache noise buffers to avoid GC pressure on repeated sweep/back calls
  private sweepBuffer: AudioBuffer | null = null;
  private backBuffer: AudioBuffer | null = null;
  private whooshBuffer: AudioBuffer | null = null;

  /** Blade sweep: filtered noise burst with pitch sweep (whoosh) */
  sweep() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.35;

    // Reuse cached noise buffer to avoid per-call allocation
    if (!this.sweepBuffer) {
      const bufferSize = ctx.sampleRate * duration;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
      }
      this.sweepBuffer = buffer;
    }

    const noise = ctx.createBufferSource();
    noise.buffer = this.sweepBuffer;

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

    // Reuse cached noise buffer
    if (!this.backBuffer) {
      const bufferSize = ctx.sampleRate * duration;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * (i / bufferSize);
      }
      this.backBuffer = buffer;
    }

    const noise = ctx.createBufferSource();
    noise.buffer = this.backBuffer;

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

  /** Boot: low rising tone with a soft harmonic shimmer */
  boot() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 1.15;

    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(82, now);
    osc.frequency.exponentialRampToValueAtTime(196, now + duration * 0.65);
    osc.frequency.exponentialRampToValueAtTime(392, now + duration);

    const shimmer = ctx.createOscillator();
    shimmer.type = "triangle";
    shimmer.frequency.setValueAtTime(330, now + 0.15);
    shimmer.frequency.exponentialRampToValueAtTime(660, now + duration);

    const master = ctx.createGain();
    master.gain.setValueAtTime(0, now);
    master.gain.linearRampToValueAtTime(0.12, now + 0.18);
    master.gain.setValueAtTime(0.12, now + duration * 0.72);
    master.gain.exponentialRampToValueAtTime(0.001, now + duration);

    const shimmerGain = ctx.createGain();
    shimmerGain.gain.setValueAtTime(0, now);
    shimmerGain.gain.linearRampToValueAtTime(0.035, now + 0.35);
    shimmerGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    osc.connect(master);
    shimmer.connect(shimmerGain);
    shimmerGain.connect(master);
    master.connect(ctx.destination);

    osc.start(now);
    shimmer.start(now + 0.15);
    osc.stop(now + duration);
    shimmer.stop(now + duration);
  }

  /**
   * Whoosh: deeper blade transition sound with more low-end.
   * Uses a lowpass-filtered noise burst with a sub-bass sine sweep
   * for a fuller, more cinematic transition effect.
   */
  whoosh() {
    if (this.muted) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.5;

    // Cached noise buffer for the whoosh
    if (!this.whooshBuffer) {
      const bufferSize = ctx.sampleRate * duration;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
      }
      this.whooshBuffer = buffer;
    }

    const noise = ctx.createBufferSource();
    noise.buffer = this.whooshBuffer;

    // Lowpass filter for deep, muffled whoosh (more low-end than sweep)
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(800, now);
    filter.frequency.exponentialRampToValueAtTime(120, now + duration);
    filter.Q.value = 1.2;

    // Sub-bass sine sweep for extra low-end punch
    const subBass = ctx.createOscillator();
    subBass.type = "sine";
    subBass.frequency.setValueAtTime(120, now);
    subBass.frequency.exponentialRampToValueAtTime(40, now + duration);

    // Gain envelopes
    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0, now);
    noiseGain.gain.linearRampToValueAtTime(0.18, now + 0.08);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    const bassGain = ctx.createGain();
    bassGain.gain.setValueAtTime(0, now);
    bassGain.gain.linearRampToValueAtTime(0.12, now + 0.1);
    bassGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    noise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(ctx.destination);

    subBass.connect(bassGain);
    bassGain.connect(ctx.destination);

    noise.start(now);
    subBass.start(now);
    noise.stop(now + duration);
    subBass.stop(now + duration);
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
    this.ambientNodes = { osc, lfo, gain };
  }

  stopAmbient() {
    if (this.ambientNodes) {
      try {
        this.ambientNodes.osc.stop();
      } catch {}
      try {
        this.ambientNodes.lfo.stop();
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
      case "boot": this.boot(); break;
      case "whoosh": this.whoosh(); break;
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
