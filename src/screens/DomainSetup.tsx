import { useState, useEffect, useMemo, useCallback } from "react";
import { motion } from "framer-motion";
import { apiGet, type TaxonomyData } from "../lib/api";
import { LoadingState } from "../components/LoadingState";
import { ErrorState } from "../components/ErrorState";

const DOMAIN_ICONS: Record<string, string> = {
  STEM: "🔬",
  Humanities: "📖",
  "Social Science": "🌐",
  Professional: "💼",
  Other: "📚",
};

export function DomainSetup({ onStart }: { onStart?: (subjects: string[], tier: string) => void }) {
  const [taxonomy, setTaxonomy] = useState<TaxonomyData>({ categories: {} });
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [tier, setTier] = useState("1");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  if (loading)
    return <LoadingState message="Loading taxonomy..." />;

  if (error)
    return (
      <div style={{ padding: 24 }}>
        <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Domain Setup</h1>
        <ErrorState message={`Failed to load taxonomy: ${error}`} onRetry={load} />
      </div>
    );

  return (
    <div style={{ padding: 24, overflowY: "auto", height: "100%" }}>
      <h1 style={{ fontSize: 24, marginBottom: 20, color: "var(--accent)" }}>Domain Setup</h1>

      {/* Domain sections */}
      {Object.entries(taxonomy.categories || {}).map(([domain, subjects]) => {
        const isExpanded = expanded.has(domain);
        const domainSubjects = subjects as string[];
        const selectedInDomain = domainSubjects.filter((s) => selected.has(s)).length;
        const icon = DOMAIN_ICONS[domain] || "📚";

        return (
          <div key={domain} className="panel" style={{ marginBottom: 12, overflow: "hidden" }}>
            <div
              role="button"
              tabIndex={0}
              aria-label={`${domain} (${domainSubjects.length} subjects${selectedInDomain > 0 ? `, ${selectedInDomain} selected` : ""})`}
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
                borderBottom: isExpanded ? "1px solid var(--border)" : "none",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15, fontWeight: 600 }}>
                <span>{icon}</span>
                {domain} ({domainSubjects.length})
                {selectedInDomain > 0 && (
                  <span style={{ fontSize: 11, padding: "1px 6px", borderRadius: 3, background: "var(--accent-secondary)", color: "var(--bg)" }}>
                    {selectedInDomain} selected
                  </span>
                )}
              </span>
              <span style={{ color: "var(--text-dim)" }}>{isExpanded ? "▼" : "▶"}</span>
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
                      background: selected.has(s) ? "var(--surface-raised)" : "transparent",
                      borderLeft: selected.has(s) ? "3px solid var(--accent)" : "3px solid transparent",
                    }}
                  >
                    <span style={{ color: selected.has(s) ? "var(--accent)" : "var(--text-dim)" }}>
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

      {/* Probing Depth */}
      <div className="panel" style={{ padding: 16, marginBottom: 16 }}>
        <h2 style={{ marginBottom: 10, fontSize: 14, color: "var(--accent-secondary)", textTransform: "uppercase", letterSpacing: 1 }}>
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
              background: tier === t ? "var(--surface-raised)" : "transparent",
            }}
          >
            <span style={{ color: tier === t ? "var(--accent)" : "var(--text-dim)" }}>
              {tier === t ? "✓" : "○"}
            </span>
            {label}
          </div>
        ))}
      </div>

      {/* Summary + Actions */}
      <div className="panel-raised" style={{ padding: 16, marginBottom: 16 }}>
        <div style={{ fontSize: 14, marginBottom: 8 }}>
          Selected: <span style={{ color: "var(--accent)" }}>{selected.size} subjects</span>
        </div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 12 }}>
          Estimated questions: ~{estimatedQuestions} (Tier {tier})
        </div>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <button onClick={selectAll} style={{ flex: 1, padding: 6, fontSize: 12, background: "var(--surface-raised)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer" }}>
            Select All
          </button>
          <button onClick={selectNone} style={{ flex: 1, padding: 6, fontSize: 12, background: "var(--surface-raised)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer" }}>
            Select None
          </button>
          <button onClick={expandAll} style={{ flex: 1, padding: 6, fontSize: 12, background: "var(--surface-raised)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer" }}>
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
