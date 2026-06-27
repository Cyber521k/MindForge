import { useState, useEffect, useMemo, useCallback, type CSSProperties } from "react";
import { motion } from "framer-motion";
import { apiGet, type TaxonomyData } from "../lib/api";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

const XBOX = {
  primaryText: "#FFF8DC",
  neonGreen: "var(--xbox-neon-green, #00ff41)",
  chartreuse: "var(--xbox-chartreuse, #ccff00)",
  dimGreen: "var(--xbox-dim-green, #5f8f5f)",
  glow: "var(--xbox-glow, 0 0 18px rgba(0, 255, 65, 0.45))",
};

const screenStyle: CSSProperties = {
  padding: 24,
  paddingRight: 128,
  overflowY: "auto",
  height: "100%",
  position: "relative",
  color: XBOX.primaryText,
};

const titleStyle: CSSProperties = {
  fontSize: 24,
  marginBottom: 8,
  background: "linear-gradient(180deg, #C0C0C0, #808080)",
  backgroundClip: "text",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
  color: "transparent",
  fontFamily: "'Arial Black', Impact, sans-serif",
  fontWeight: 900,
  letterSpacing: 0,
  textTransform: "uppercase",
};

const headerGlowLineStyle: CSSProperties = {
  height: 1,
  marginBottom: 20,
  background: `linear-gradient(90deg, transparent, ${XBOX.neonGreen}, transparent)`,
  opacity: 0.6,
};

const xboxPanelStyle: CSSProperties = {
  clipPath: "polygon(12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%, 0 12px)",
  border: `1px solid ${XBOX.neonGreen}`,
  boxShadow: XBOX.glow,
  background: "rgba(10, 26, 10, 0.75)",
  backdropFilter: "blur(8px)",
  WebkitBackdropFilter: "blur(8px)",
  color: XBOX.primaryText,
};

const sectionHeadingStyle: CSSProperties = {
  marginBottom: 10,
  fontSize: 14,
  color: XBOX.neonGreen,
  textTransform: "uppercase",
  letterSpacing: 0,
};

const decorativeIconStyle: CSSProperties = {
  position: "absolute",
  top: 24,
  right: 24,
  width: 80,
  height: 80,
  filter: `drop-shadow(${XBOX.glow})`,
  pointerEvents: "none",
};

function DomainSetupIcon() {
  return (
    <div aria-hidden="true" style={decorativeIconStyle}>
      <div
        style={{
          position: "absolute",
          inset: 8,
          borderRadius: "50%",
          border: `2px solid ${XBOX.neonGreen}`,
          transform: "perspective(220px) rotateY(-24deg) rotateX(10deg)",
          boxShadow: `inset 0 0 20px rgba(0, 255, 65, 0.18), ${XBOX.glow}`,
        }}
      >
        <div style={{ position: "absolute", top: "49%", left: 4, right: 4, height: 2, background: XBOX.chartreuse, opacity: 0.75 }} />
        <div style={{ position: "absolute", top: 10, bottom: 10, left: "49%", width: 2, background: XBOX.chartreuse, opacity: 0.75 }} />
        <div style={{ position: "absolute", inset: "12px 20px", borderLeft: `2px solid ${XBOX.neonGreen}`, borderRight: `2px solid ${XBOX.neonGreen}`, borderRadius: "50%" }} />
        <div style={{ position: "absolute", inset: "20px 8px", borderTop: `2px solid ${XBOX.neonGreen}`, borderBottom: `2px solid ${XBOX.neonGreen}`, borderRadius: "50%" }} />
      </div>
    </div>
  );
}

function ScreenHeader() {
  return (
    <>
      <DomainSetupIcon />
      <h1 style={titleStyle}>Domain Setup</h1>
      <div style={headerGlowLineStyle} />
    </>
  );
}

const DOMAIN_ICONS: Record<string, string> = {
  STEM: "🔬",
  Humanities: "📖",
  "Social Science": "🌐",
  Professional: "💼",
  Other: "📚",
  Agent_Frameworks: "🤖",
  Programming_Languages: "💻",
  Blockchain_Web3: "🔗",
  DevOps_Infrastructure: "🖥",
  Security_Cryptography: "🛡",
};

export function DomainSetup({ onStart }: { onStart?: (subjects: string[], tier: string) => void }) {
  const [taxonomy, setTaxonomy] = useState<TaxonomyData>({ categories: {} });
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [tier, setTier] = useState("1");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    apiGet<TaxonomyData>("/api/taxonomy")
      .then((data) => {
        setTaxonomy(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err?.message || String(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const toggle = (subject: string) => {
    const s = new Set(selected);
    if (s.has(subject)) s.delete(subject);
    else s.add(subject);
    setSelected(s);
  };

  const toggleExpand = (domain: string) => {
    const e = new Set(expanded);
    if (e.has(domain)) e.delete(domain);
    else e.add(domain);
    setExpanded(e);
  };

  const selectAll = () => {
    const all = new Set<string>();
    Object.entries(taxonomy.categories || {}).forEach(([, subjects]) => {
      (subjects as string[]).forEach((s) => all.add(s));
    });
    setSelected(all);
  };

  const selectNone = () => setSelected(new Set());

  const expandAll = () => setExpanded(new Set(Object.keys(taxonomy.categories || {})));

  const estimatedQuestions = useMemo(() => {
    return selected.size * 25 * (tier === "1" ? 1 : tier === "2" ? 2 : 3);
  }, [selected, tier]);

  // Filter domains based on search query
  const filteredCategories = useMemo(() => {
    const cats = taxonomy.categories || {};
    if (!searchQuery.trim()) return cats;

    const filtered: Record<string, string[]> = {};
    const q = searchQuery.toLowerCase();
    for (const [domain, subjects] of Object.entries(cats)) {
      const matched = (subjects as string[]).filter((s) =>
        s.toLowerCase().includes(q) || domain.toLowerCase().includes(q)
      );
      if (matched.length > 0) {
        filtered[domain] = matched;
      }
    }
    return filtered;
  }, [taxonomy.categories, searchQuery]);

  const totalSubjects = useMemo(() => {
    return Object.values(taxonomy.categories || {}).reduce(
      (sum, subjects) => sum + (subjects as string[]).length,
      0
    );
  }, [taxonomy.categories]);

  if (loading)
    return <LoadingState message="Loading taxonomy..." />;

  if (error)
    return (
      <div style={screenStyle}>
        <ScreenHeader />
        <ErrorState message={`Failed to load taxonomy: ${error}`} onRetry={load} />
      </div>
    );

  return (
    <div style={screenStyle}>
      <ScreenHeader />

      {/* Search filter */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 16, marginBottom: 16 }}>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search subjects across all domains..."
          aria-label="Filter subjects"
          style={{
            width: "100%",
            padding: "10px 14px",
            background: "rgba(0, 0, 0, 0.24)",
            border: `1px solid ${XBOX.neonGreen}`,
            borderRadius: 6,
            color: XBOX.primaryText,
            fontSize: 14,
          }}
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery("")}
            style={{
              position: "absolute",
              right: 38,
              marginTop: -34,
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontSize: 16,
              color: XBOX.dimGreen,
            }}
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {/* Domain sections */}
      {Object.entries(filteredCategories).map(([domain, subjects]) => {
        const isExpanded = expanded.has(domain) || !!searchQuery.trim();
        const domainSubjects = subjects as string[];
        const selectedInDomain = domainSubjects.filter((s) => selected.has(s)).length;
        const icon = DOMAIN_ICONS[domain] || "📚";

        return (
          <div key={domain} className="panel" style={{ ...xboxPanelStyle, marginBottom: 12, overflow: "hidden" }}>
            <div
              role="button"
              tabIndex={0}
              aria-label={`${domain.replace(/_/g, " ")} (${domainSubjects.length} subjects${selectedInDomain > 0 ? `, ${selectedInDomain} selected` : ""})`}
              aria-expanded={isExpanded}
              onClick={() => toggleExpand(domain)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  toggleExpand(domain);
                }
              }}
              style={{
                padding: "12px 16px",
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                color: XBOX.primaryText,
                background: selectedInDomain > 0 ? "rgba(204, 255, 0, 0.15)" : "transparent",
                borderBottom: isExpanded ? `1px solid ${XBOX.neonGreen}` : "none",
                borderLeft: selectedInDomain > 0 ? `3px solid ${XBOX.chartreuse}` : "3px solid transparent",
                boxShadow: selectedInDomain > 0 ? XBOX.glow : "none",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15, fontWeight: 600 }}>
                <span aria-hidden="true">{icon}</span>
                {domain.replace(/_/g, " ")}
                {/* Subject count badge */}
                <span style={{
                  fontSize: 11,
                  padding: "1px 8px",
                  borderRadius: 10,
                  background: "rgba(204, 255, 0, 0.12)",
                  color: XBOX.primaryText,
                  fontWeight: 500,
                }}>
                  {domainSubjects.length}
                </span>
                {selectedInDomain > 0 && (
                  <span style={{ fontSize: 11, padding: "1px 6px", borderRadius: 3, background: XBOX.chartreuse, color: "#001f08" }}>
                    {selectedInDomain} selected
                  </span>
                )}
              </span>
              <span style={{ color: XBOX.dimGreen }} aria-hidden="true">{isExpanded ? "▼" : "▶"}</span>
            </div>
            {isExpanded && (
              <div style={{ padding: "8px 16px" }}>
                {domainSubjects.map((s) => (
                  <motion.div
                    key={s}
                    role="button"
                    tabIndex={0}
                    aria-label={`${s.replace(/_/g, " ")}${selected.has(s) ? " (selected)" : ""}`}
                    aria-pressed={selected.has(s)}
                    whileHover={{ x: 4 }}
                    onClick={() => toggle(s)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggle(s);
                      }
                    }}
                    style={{
                      padding: "6px 10px",
                      cursor: "pointer",
                      borderRadius: 4,
                      fontSize: 13,
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      color: selected.has(s) ? XBOX.chartreuse : XBOX.primaryText,
                      background: selected.has(s) ? "rgba(204, 255, 0, 0.15)" : "transparent",
                      borderLeft: selected.has(s) ? `3px solid ${XBOX.chartreuse}` : "3px solid transparent",
                      boxShadow: selected.has(s) ? XBOX.glow : "none",
                    }}
                  >
                    <span style={{ color: selected.has(s) ? XBOX.chartreuse : XBOX.dimGreen }} aria-hidden="true">
                      {selected.has(s) ? "✓" : "○"}
                    </span>
                    {s.replace(/_/g, " ")}
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {Object.keys(filteredCategories).length === 0 && (
        <div className="panel" style={{ ...xboxPanelStyle, padding: 24, textAlign: "center", color: XBOX.dimGreen, fontSize: 14 }}>
          No subjects match "{searchQuery}"
        </div>
      )}

      {/* Probing Depth */}
      <div className="panel" style={{ ...xboxPanelStyle, padding: 16, marginBottom: 16 }}>
        <h2 style={sectionHeadingStyle}>
          Probing Depth
        </h2>
        {[
          { t: "1", label: "Tier 1 — Breadth (one question per sub-topic)" },
          { t: "2", label: "Tier 2 — Depth (follow-up drilling)" },
          { t: "3", label: "Tier 3 — Edge Cases (adversarial / tricks)" },
        ].map(({ t, label }) => (
          <div
            key={t}
            role="button"
            tabIndex={0}
            aria-label={label}
            aria-pressed={tier === t}
            onClick={() => setTier(t)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setTier(t);
              }
            }}
            style={{
              padding: "6px 10px",
              cursor: "pointer",
              borderRadius: 4,
              fontSize: 13,
              display: "flex",
              alignItems: "center",
              gap: 8,
              color: tier === t ? XBOX.chartreuse : XBOX.primaryText,
              background: tier === t ? "rgba(204, 255, 0, 0.15)" : "transparent",
              borderLeft: tier === t ? `3px solid ${XBOX.chartreuse}` : "3px solid transparent",
              boxShadow: tier === t ? XBOX.glow : "none",
            }}
          >
            <span style={{ color: tier === t ? XBOX.chartreuse : XBOX.dimGreen }} aria-hidden="true">
              {tier === t ? "✓" : "○"}
            </span>
            {label}
          </div>
        ))}
      </div>

      {/* Summary + Actions */}
      <div className="panel-raised" style={{ ...xboxPanelStyle, padding: 16, marginBottom: 16 }}>
        <div style={{ fontSize: 14, marginBottom: 8 }}>
          Selected: <span style={{ color: XBOX.chartreuse }}>{selected.size}</span> / {totalSubjects} subjects
        </div>
        <div style={{ fontSize: 13, color: XBOX.dimGreen, marginBottom: 12 }}>
          Estimated questions: ~{estimatedQuestions} (Tier {tier})
        </div>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <button onClick={selectAll} style={{ flex: 1, padding: 6, fontSize: 12, background: "rgba(10, 26, 10, 0.7)", color: XBOX.primaryText, border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4, cursor: "pointer" }}>
            Select All
          </button>
          <button onClick={selectNone} style={{ flex: 1, padding: 6, fontSize: 12, background: "rgba(10, 26, 10, 0.7)", color: XBOX.primaryText, border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4, cursor: "pointer" }}>
            Select None
          </button>
          <button onClick={expandAll} style={{ flex: 1, padding: 6, fontSize: 12, background: "rgba(10, 26, 10, 0.7)", color: XBOX.primaryText, border: `1px solid ${XBOX.neonGreen}`, borderRadius: 4, cursor: "pointer" }}>
            Expand All
          </button>
        </div>
        <button
          className="btn-gold gold-glow"
          disabled={selected.size === 0}
          onClick={() => onStart?.(Array.from(selected), tier)}
          style={{ width: "100%", padding: 12, fontSize: 16, opacity: selected.size === 0 ? 0.4 : 1 }}
        >
          ► Start Probing ({selected.size} subjects)
        </button>
      </div>
    </div>
  );
}
