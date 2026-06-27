import { useRef, useCallback, useState, useEffect } from "react";

type SoundType = "sweep" | "select" | "scroll" | "back" | "boot" | "whoosh" | "hover" | "error";

type AmbientVoice = {
  osc: OscillatorNode;
  gain: GainNode;
};

type AmbientNodes = {
  voices: AmbientVoice[];
  filter: BiquadFilterNode;
  gain: GainNode;
  lfo: OscillatorNode;
  lfoGain: GainNode;
};

/**
 * Web Audio API sound effect manager for Xbox Blades UI.
 * Generates all sounds programmatically — no audio files needed.
 */
export class SoundEngine {
  private ctx: AudioContext | null = null;
  private muted = false;
  private ambientNodes: AmbientNodes | null = null;

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

  private prefersReducedMotion(): boolean {
    try {
      return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    } catch {
      return false;
    }
  }

  private canPlay(): boolean {
    return !this.muted && !this.prefersReducedMotion();
  }

  private createNoiseBuffer(ctx: AudioContext, duration: number, taper: "none" | "fade-in" | "fade-out" = "none") {
    const bufferSize = Math.max(1, Math.floor(ctx.sampleRate * duration));
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      let amp = 1;
      if (taper === "fade-in") amp = i / bufferSize;
      if (taper === "fade-out") amp = 1 - i / bufferSize;
      data[i] = (Math.random() * 2 - 1) * amp;
    }

    return buffer;
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
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.35;

    if (!this.sweepBuffer) {
      this.sweepBuffer = this.createNoiseBuffer(ctx, duration, "fade-out");
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

  /** Menu select: dual harmonic chime with a short noisy onset. */
  select() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.15;
    const glideEnd = now + 0.13;

    const voices = [
      { frequency: 880, endFrequency: 730, peak: 0.25 },
      { frequency: 1760, endFrequency: 1610, peak: 0.12 },
    ];

    for (const voice of voices) {
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.setValueAtTime(voice.frequency, now);
      osc.frequency.linearRampToValueAtTime(voice.endFrequency, glideEnd);

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(voice.peak, now + 0.003);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.092);

      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + duration);
    }

    const noise = ctx.createBufferSource();
    noise.buffer = this.createNoiseBuffer(ctx, 0.008);

    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.04, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.008);

    noise.connect(noiseGain);
    noiseGain.connect(ctx.destination);
    noise.start(now);
    noise.stop(now + 0.008);
  }

  /** Menu scroll: descending sine tick through a lowpass filter. */
  scroll() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.12;

    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(2200, now);
    osc.frequency.exponentialRampToValueAtTime(1200, now + duration);

    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(6000, now);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.3, now + 0.003);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.103);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + duration);
  }

  /** Back: reverse whoosh (pitch sweeps down) */
  back() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.3;

    if (!this.backBuffer) {
      this.backBuffer = this.createNoiseBuffer(ctx, duration, "fade-in");
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

  /** Boot: Xbox 2001-style harmonic startup chord. */
  boot() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 2.2;
    const attack = 0.02;
    const decay = 0.8;
    const release = 0.4;
    const sustainStart = now + duration - release;
    const peakAmplitude = 0.5;
    const harmonics = [
      { frequency: 27.3, relativeGain: 1.0 },
      { frequency: 50.8, relativeGain: 0.42 },
      { frequency: 74.2, relativeGain: 0.30 },
      { frequency: 160.2, relativeGain: 0.27 },
      { frequency: 972.7, relativeGain: 0.16 },
    ];

    for (const harmonic of harmonics) {
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.setValueAtTime(harmonic.frequency, now);

      const gain = ctx.createGain();
      const peak = peakAmplitude * harmonic.relativeGain;
      const sustain = Math.max(0.001, peak * 0.36);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(peak, now + attack);
      gain.gain.exponentialRampToValueAtTime(sustain, now + attack + decay);
      gain.gain.setValueAtTime(sustain, sustainStart);
      gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + duration);
    }
  }

  /** Transition whoosh: bandpass white-noise sweep. */
  whoosh() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.4;

    if (!this.whooshBuffer) {
      this.whooshBuffer = this.createNoiseBuffer(ctx, duration);
    }

    const noise = ctx.createBufferSource();
    noise.buffer = this.whooshBuffer;

    const filter = ctx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.setValueAtTime(200, now);
    filter.frequency.linearRampToValueAtTime(900, now + 0.2);
    filter.frequency.linearRampToValueAtTime(200, now + duration);
    filter.Q.value = 2;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.22, now + 0.05);
    gain.gain.linearRampToValueAtTime(0.08, now + 0.25);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    noise.start(now);
    noise.stop(now + duration);
  }

  /** Blade hover: very quiet high tick. */
  hover() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.03;
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(1800, now);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.08, now + 0.002);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + duration);
  }

  /** Error state: low square-wave buzz. */
  error() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx) return;

    const now = ctx.currentTime;
    const duration = 0.2;
    const osc = ctx.createOscillator();
    osc.type = "square";
    osc.frequency.setValueAtTime(200, now);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.2, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + duration);
  }

  /** Layered ambient drone with low harmonics and slow tremolo. */
  startAmbient() {
    if (!this.canPlay()) return;
    const ctx = this.ensureCtx();
    if (!ctx || this.ambientNodes) return;

    const now = ctx.currentTime;
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(250, now);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.04, now + 1);

    const lfo = ctx.createOscillator();
    lfo.type = "sine";
    lfo.frequency.setValueAtTime(0.7, now);

    const lfoGain = ctx.createGain();
    lfoGain.gain.setValueAtTime(0.02, now);
    lfo.connect(lfoGain);
    lfoGain.connect(gain.gain);

    const voices = [
      { frequency: 27.3, type: "sine" as OscillatorType, level: 0.65 },
      { frequency: 50.8, type: "sawtooth" as OscillatorType, level: 0.24 },
      { frequency: 74.2, type: "sawtooth" as OscillatorType, level: 0.16 },
    ].map((voice) => {
      const osc = ctx.createOscillator();
      osc.type = voice.type;
      osc.frequency.setValueAtTime(voice.frequency, now);

      const voiceGain = ctx.createGain();
      voiceGain.gain.setValueAtTime(voice.level, now);

      osc.connect(voiceGain);
      voiceGain.connect(filter);
      osc.start(now);

      return { osc, gain: voiceGain };
    });

    filter.connect(gain);
    gain.connect(ctx.destination);
    lfo.start(now);

    this.ambientNodes = { voices, filter, gain, lfo, lfoGain };
  }

  stopAmbient() {
    if (!this.ambientNodes || !this.ctx) return;

    const nodes = this.ambientNodes;
    const now = this.ctx.currentTime;
    const release = 5;
    this.ambientNodes = null;

    try {
      nodes.gain.gain.cancelAndHoldAtTime(now);
    } catch {
      nodes.gain.gain.cancelScheduledValues(now);
      nodes.gain.gain.setValueAtTime(0.04, now);
    }
    nodes.gain.gain.linearRampToValueAtTime(0.0001, now + release);

    for (const voice of nodes.voices) {
      try {
        voice.osc.stop(now + release);
      } catch {}
    }

    try {
      nodes.lfo.stop(now + release);
    } catch {}

    window.setTimeout(() => {
      for (const voice of nodes.voices) {
        try {
          voice.osc.disconnect();
          voice.gain.disconnect();
        } catch {}
      }
      try {
        nodes.lfo.disconnect();
        nodes.lfoGain.disconnect();
        nodes.filter.disconnect();
        nodes.gain.disconnect();
      } catch {}
    }, release * 1000 + 100);
  }

  play(type: SoundType) {
    switch (type) {
      case "sweep": this.sweep(); break;
      case "select": this.select(); break;
      case "scroll": this.scroll(); break;
      case "back": this.back(); break;
      case "boot": this.boot(); break;
      case "whoosh": this.whoosh(); break;
      case "hover": this.hover(); break;
      case "error": this.error(); break;
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
