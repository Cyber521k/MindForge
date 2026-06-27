# Multi-Model Ensemble Approaches for LLM Output Verification & Cross-Checking

> **Research document for MindForge** — covers ensemble judge strategies, model diversity, cost-quality tradeoffs, disagreement resolution, cascade verification, and practical architectures compatible with MindForge's adapter system (OpenAI, OpenRouter, Ollama, MLX, Exo).
>
> **Date:** June 27, 2026  
> **Author:** Hermes Agent (research synthesis)  
> **Status:** Research / design reference

---

## Table of Contents

1. [Multi-LLM Judges: Combining Verdicts](#1-multi-llm-judges-combining-verdicts)
2. [Model Diversity: Why Different Families Matter](#2-model-diversity-why-different-families-matter)
3. [Cost-Quality Tradeoffs](#3-cost-quality-tradeoffs)
4. [Disagreement Resolution & Escalation](#4-disagreement-resolution--escalation)
5. [Cascade Verification](#5-cascade-verification)
6. [Practical Architecture for MindForge](#6-practical-architecture-for-mindforge)
7. [Key References](#7-key-references)

---

## 1. Multi-LLM Judges: Combining Verdicts

### Core Concept

Instead of relying on a **single LLM judge** (e.g., GPT-4o alone), use a **panel of multiple LLM judges** that each independently evaluate the same output, then **aggregate their verdicts** into a final decision. This is called:

- **PoLL** (Panel of LLM evaluators) — Verga et al., 2024
- **LLM-as-a-Jury** — Arize AI, 2025
- **LLM Jury** — Comet, 2025

The key insight from classical ML ensemble theory: combining weak learners reduces **both bias and variance** — but only if the learners are **diverse** (their errors point in different directions).

### Verdict Aggregation Methods

#### 🔢 Majority Vote (Plurality)

- **Each judge outputs a verdict** (correct/incorrect, accept/reject, A/B/TIE)
- **Final verdict = most common answer** among N judges
- **Odd-numbered panels** (3, 5, 7) avoid ties
- **Simplest to implement** — no weight calibration needed
- **Best when judges have similar accuracy** and you have no calibration data

```
Judge 1: ACCEPT   (conf: 0.9)
Judge 2: REJECT   (conf: 0.6)
Judge 3: ACCEPT   (conf: 0.85)
→ Majority: ACCEPT (2 vs 1)
```

**Limitation:** Treats all judges equally. A weak model's vote counts the same as a strong model's.

#### ⚖️ Weighted Vote (Confidence-Weighted)

- **Each judge outputs a verdict + confidence score**
- **Final verdict = weighted sum** of verdicts, weighted by confidence
- **Weights can be:**
  - **Self-reported confidence** from the judge (simple but poorly calibrated)
  - **Calibrated confidence** via logit-based calibration (Amazon Science, 2024 — reduces calibration error by 46%)
  - **Historical accuracy** per judge on a labeled validation set (most reliable but requires calibration data)
  - **Domain-specific weights** — e.g., a model strong at math gets higher weight on STEM entries

```
Judge 1 (GPT-4o):     ACCEPT, conf=0.92  → weight = 0.92
Judge 2 (Claude):     REJECT, conf=0.65  → weight = 0.65
Judge 3 (Gemini):     ACCEPT, conf=0.80  → weight = 0.80

Weighted ACCEPT:  (0.92 + 0.80) = 1.72
Weighted REJECT:  (0.65)        = 0.65
→ Weighted verdict: ACCEPT (1.72 vs 0.65)
```

**Key research:** Amazon Science (Hovsepian et al., 2024) designed a **cost-aware cascading LLM ensemble policy** using calibrated confidence scores that achieves improved accuracy while reducing inference cost by **>2× vs. conventional weighted majority voting**.

#### 🤝 Consensus (Unanimous / Supermajority)

- **Unanimous consensus:** All judges must agree. Very conservative — high precision, low throughput.
- **Supermajority consensus:** Require ≥ 2/3 or ≥ 3/4 agreement. Balances precision and coverage.
- **Conservative consensus** (W&B): If any judge disagrees, flag for review rather than auto-accepting.

```
3 judges, all say ACCEPT  → Unanimous: ACCEPT (auto-approve)
3 judges, 2 ACCEPT 1 REJECT → Supermajority (2/3): ACCEPT
3 judges, 1 ACCEPT 2 REJECT → Supermajority fails → ESCALATE to human/expensive model
```

**When to use which:**

| Method | Best For | Precision | Throughput | Cost |
|--------|----------|-----------|------------|------|
| **Majority vote** | Equal-strength judges, no calibration data | Medium | High | Medium |
| **Weighted vote** | Judges with known accuracy differences | High | High | Medium |
| **Unanimous** | High-stakes decisions (safety, medical) | Very High | Low | Medium |
| **Supermajority (2/3)** | Balanced approach for most use cases | High | Medium | Medium |

### Research-Backed Findings

- **PoLL (Verga et al., 2024):** A panel of smaller, diverse models **outperforms a single large judge** (GPT-4) across 6 datasets while being **7× cheaper**. The panel exhibits less intra-model bias because it's composed of disjoint model families.
- **LLM-as-a-Jury (Arize, 2025):** Meta-Judge pipelines with 3-stage juror selection deliver **+15% reliability** vs. single judges. Diverse juror pools capture under-represented perspectives.
- **Self-consistency (Wang et al., ICLR 2023):** Aggregating reasoning paths from the same model provides **10–20 point accuracy lifts** — even without model diversity.
- **Optimal panel size:** Research on multi-annotator agreement (Krippendorff's alpha, Fleiss' kappa) shows the ensemble effect is strongest with **3–5 annotators**. Beyond that, marginal gains plateau while costs rise linearly.

---

## 2. Model Diversity: Why Different Families Matter

### The Self-Preference Bias Problem

**LLMs systematically prefer outputs from their own model family.** This is not random noise — it's a **systematic, repeatable bias**.

**Key findings (Yang et al., 2026; arXiv 2410.21819, NeurIPS 2024):**

- Judges **over-rate outputs from their own model family** by a measurable margin
- LLMs assign higher evaluations to outputs with **lower perplexity** than human evaluators, regardless of whether the outputs were self-generated
- On IFEval (programmatically verifiable rubrics), judges are **up to 50% more likely** to incorrectly mark their own family's failed outputs as passing
- Self-preference operates at the **stylistic/linguistic level** — a model recognizes and rewards its own "voice"

### Why Cross-Family Diversity Fixes This

| Approach | Problem |
|----------|---------|
| **Same family, different model** (GPT-4o judging GPT-4o-mini) | Shared training data, tokenizer, RLHF style → correlated biases |
| **Same model, multiple runs** (GPT-4o × 3) | Repeated trial, **not** an ensemble. Same systematic biases, just averaged noise |
| **Different families** (OpenAI + Anthropic + Google) | Disjoint training corpora, different RLHF approaches, different tokenizers → biases point in different directions → **cancel out on aggregation** |

**The ensemble principle:** Two independent judges might both err, but if their mistakes point in *different directions*, majority voting or disagreement inspection becomes a powerful signal. Three correlated judges = one judge with 3× the requests.

### Documented LLM Judge Biases (Full Catalog)

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Self-preference** | Over-rates own family's outputs | Cross-family panel |
| **Length/verbosity** | Rewards longer answers regardless of correctness | Normalize by length, or use rubric with explicit length criteria |
| **Position bias** | In A-vs-B judging, order outranks content | Swap order, run both, aggregate |
| **Stylistic bias** | Favors argumentative structures matching training distribution | Cross-family panel |
| **Conservative bias** | Defaults to "unsure"/"tie" on tricky inputs even when answer is clear | Use direct scoring instead of pairwise |
| **Confidence bias** | Rewards confident-sounding answers; penalizes "I don't know" even when correct | Penalize overconfidence in rubric |
| **Specialist gaps** | Can't reliably judge medical, legal, or technical answers | Domain-specific judge selection |

### Practical Diversity Configurations for MindForge

MindForge's adapter system already supports **4 different backend families**:

| Adapter | Model Family | Typical Models | Cost Tier |
|---------|-------------|----------------|-----------|
| `OpenAIAdapter` | OpenAI (GPT) | GPT-4o, GPT-4o-mini, o3 | $$$ / $ |
| `OpenRouterAdapter` | Anthropic / Google / Meta / Mistral | Claude, Gemini, Llama, Mixtral | $$ - $$$ |
| `OllamaAdapter` | Local GGUF (Meta, Mistral, Google) | Llama, Qwen, Gemma, Mistral | Free (local) |
| `MLXAdapter` | Local MLX (Meta, Mistral) | Llama-3.2-3B, Mistral-7B | Free (local) |

**Recommended diverse panels:**

- **Budget panel (free + cheap):** Ollama/Llama-3 + MLX/Llama-3.2 + GPT-4o-mini → 3 families, ~$0.15/M tokens for the paid member
- **Balanced panel:** GPT-4o-mini + Claude Haiku (via OpenRouter) + Gemini Flash (via OpenRouter) → 3 frontier families, all cheap tier
- **High-accuracy panel:** GPT-4o + Claude Sonnet (via OpenRouter) + Gemini Pro (via OpenRouter) → 3 frontier families, premium tier
- **Max diversity (5 judges):** GPT-4o-mini + Claude Haiku + Gemini Flash + Ollama/Llama-3 + MLX/Llama-3.2 → 4 families + 1 local redundancy

---

## 3. Cost-Quality Tradeoffs

### The 5–25× Cost Ratio

Current model families offer **5–25× cost ratios** between efficient and frontier tiers:

| Tier | Example Models | Approx. Cost (input/output per M tokens) | Use in Verification |
|------|----------------|------------------------------------------|---------------------|
| **Ultra-cheap** | GPT-4o-mini, Gemini Flash-Lite | $0.15 / $0.60 | First-pass filter, easy cases |
| **Cheap** | Claude Haiku 4.5, Gemini Flash | $1.00 / $5.00 | First-pass judge, medium cases |
| **Mid** | GPT-4o, Claude Sonnet | $2.50 / $10.00 | Second-pass judge, escalated cases |
| **Premium** | GPT-5, Claude Opus, Gemini Pro | $5.00 / $25.00+ | Final arbiter, hard cases only |
| **Free (local)** | Ollama models, MLX models | $0 (hardware cost only) | Bulk filtering, offline batches |

### When to Use Each Tier in the Verification Pipeline

#### 🟢 Cheap Models (GPT-4o-mini, Haiku) — First Line

- **Filter easy cases:** Obvious correct/incorrect answers that don't need deep reasoning
- **High-confidence pass-through:** If a cheap model judges with confidence > 0.9, skip expensive models
- **Bulk batch processing:** Review 100s/1000s of training entries at low cost
- **Classification tasks:** Binary accept/reject is easier than generation — cheap models handle it well
- **Research finding:** PoLL showed that a panel of smaller models **outperforms** a single GPT-4 judge at **1/7th the cost**

#### 🟡 Mid-Tier Models (GPT-4o, Sonnet) — Disagreement Resolution

- **Break ties:** When cheap judges disagree, escalate to mid-tier
- **Cross-verify uncertain cases:** Confidence between 0.5–0.8 from the cheap layer
- **Domain-specific judging:** Use the strongest model for the subject domain (e.g., GPT-4o for code, Claude for reasoning)

#### 🔴 Premium Models (GPT-5, Claude Opus) — Final Arbiter

- **Use sparingly:** Only for cases where mid-tier judges also disagree or confidence remains low
- **Safety-critical entries:** Medical, legal, security-related subjects
- **Novel/ambiguous cases:** Questions the pipeline hasn't seen before (routing cold start)
- **Estimated usage:** <5% of total entries if the cascade is well-tuned

### Cost Calculation Example (MindForge Batch Review)

**Scenario:** Review 1,000 training entries

| Strategy | Model(s) | Calls | Cost per Call (est.) | Total Cost |
|----------|----------|-------|---------------------|------------|
| **Single GPT-4o judge** | GPT-4o | 1,000 | ~$0.01 | ~$10.00 |
| **PoLL (3 cheap models)** | GPT-4o-mini + Haiku + Flash | 3,000 | ~$0.002 each | ~$6.00 |
| **Cascade (cheap → mid → premium)** | GPT-4o-mini (900) → GPT-4o (80) → Opus (20) | 1,000 | Varies | ~$3.50 |
| **PoLL + Cascade hybrid** | 3 cheap (700) → 2 mid (250) → 1 premium (50) | 1,000 | Varies | ~$5.00 |

**Key insight from RouteLLM research:** Proper routing maintains **95% of frontier model quality** while routing **85% of queries to cheaper models**, achieving **45–85% cost reductions** depending on workload.

### The Self-Consistency Free Win

Before spending money on multiple different models, consider **running the same cheap model multiple times** with different temperature settings:

- **Self-consistency (Wang et al., 2023):** Sample N responses from the same model at temperature 0.7+, take the majority answer
- **Cost:** N × cost of one cheap model call (still very cheap)
- **Accuracy lift:** 10–20 points on reasoning tasks
- **Limitation:** Does not address systematic biases — only reduces variance

**Best practice:** Use self-consistency as a **cheap pre-filter** — if a single model is highly confident and consistent across 3 samples, skip the panel entirely.

---

## 4. Disagreement Resolution & Escalation

### Core Principle: Disagreement Is Data

> When judges disagree, you've found an **ambiguous case worth human review**. The disagreement set is the panel's real product — not the aggregated verdict.
>
> — orq.ai, "Weak Judges, Strong Panel"

A well-functioning panel doesn't try to force consensus on hard cases. It **routes those cases to escalation**.

### Disagreement Scenarios & Resolution Strategies

#### Scenario 1: Majority Agreement, One Dissenter

```
Judge 1: ACCEPT (conf: 0.9)
Judge 2: ACCEPT (conf: 0.85)
Judge 3: REJECT (conf: 0.7)
```

**Resolution options:**
- **Trust the majority** (accept) — but flag the dissent for audit
- **Weight by confidence:** Accept votes have higher total confidence → accept
- **Escalate if dissenter is high-confidence:** If Judge 3's confidence was 0.95, the dissent is a strong signal → escalate

#### Scenario 2: Split Decision (no majority)

```
Judge 1: ACCEPT (conf: 0.6)
Judge 2: REJECT (conf: 0.55)
Judge 3: TIE/UNSURE (conf: 0.3)
```

**Resolution:** Always escalate. No signal here — the case is genuinely ambiguous.

#### Scenario 3: All Agree (Unanimous)

```
Judge 1: ACCEPT (conf: 0.95)
Judge 2: ACCEPT (conf: 0.90)
Judge 3: ACCEPT (conf: 0.92)
```

**Resolution:** Auto-accept. No escalation needed. This should be the majority of cases (~70-85%).

#### Scenario 4: High Confidence Disagreement

```
Judge 1: ACCEPT  (conf: 0.95)
Judge 2: REJECT   (conf: 0.90)
Judge 3: ACCEPT   (conf: 0.88)
```

**Resolution:** Escalate despite majority. When two judges are both highly confident but disagree, the question is likely **edge-case or domain-specific** — exactly where a stronger model (or human) adds the most value.

### Escalation Ladder

```
┌─────────────────────────────────────────────────────────┐
│                    ESCALATION LADDER                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  TIER 0: Self-consistency check                         │
│  └─ Same cheap model × 3 samples → majority             │
│     └─ If unanimous + high conf → DONE (auto-accept)    │
│     └─ If disagreement → escalate to Tier 1              │
│                                                         │
│  TIER 1: Cheap panel (3 diverse judges)                 │
│  └─ GPT-4o-mini + Haiku + Flash → majority/weighted     │
│     └─ If 2/3+ agree + avg conf > 0.8 → DONE            │
│     └─ If split or low confidence → escalate to Tier 2   │
│                                                         │
│  TIER 2: Mid-tier judge (single strong model)           │
│  └─ GPT-4o or Claude Sonnet → decisive verdict          │
│     └─ If conf > 0.85 → DONE                            │
│     └─ If conf < 0.85 or contradicts Tier 1 → Tier 3   │
│                                                         │
│  TIER 3: Premium judge + web search                     │
│  └─ GPT-5 / Claude Opus + DuckDuckGo search              │
│     └─ Cross-reference web-sourced answer                │
│     └─ If conf > 0.85 → DONE                            │
│     └─ If still uncertain → Tier 4                      │
│                                                         │
│  TIER 4: Human review queue                             │
│  └─ Flag entry for manual review                        │
│  └─ Store all judge verdicts + reasoning for context     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Disagreement Metrics

Track these to tune the pipeline:

| Metric | Formula | Target |
|--------|---------|--------|
| **Agreement rate** | % of entries where all judges agree | 70–85% |
| **Escalation rate** | % of entries escalated to higher tier | 5–15% |
| **Human review rate** | % of entries reaching Tier 4 | <5% |
| **Cohen's κ** (pairwise) | Agreement between two judges, corrected for chance | >0.6 = good |
| **Fleiss' κ** (multi-judge) | Agreement across all judges | >0.6 = good |
| **Flip rate** (comparative) | % of pairwise verdicts that change on prompt tweaks | <10% = stable |

**Critical stability finding (Wang et al., 2025):** Comparative/pairwise verdicts flip **~35%** on tiny prompt tweaks vs. **~9%** for direct scores. This ~4× instability is why **direct scoring (pointwise)** is preferred for ensemble verification — it's more stable across judges.

---

## 5. Cascade Verification

### What Is a Cascade?

A **cascade** is a sequential routing architecture where requests are processed by a **cheap model first**, and only **escalated to more expensive models** when the cheap model's confidence is low.

```
Query ──→ [Cheap Model] ──(high conf?)──→ DONE
                         │
                         └─(low conf)──→ [Mid Model] ──(high conf?)──→ DONE
                                              │
                                              └─(low conf)──→ [Premium Model] ──→ DONE
```

### Key Distinction: Routing vs. Cascading

| | **Routing** | **Cascading** |
|---|---|---|
| **Decision** | One-shot, upfront classification | Sequential escalation |
| **Execution** | One model handles the query | Starts cheap, escalates if confidence low |
| **Latency cost** | Classifier adds latency before generation | Each escalation adds another model call |
| **Key requirement** | Accurate upfront classifier | Good confidence calibration |
| **Best for** | Known query categories | Unknown difficulty distribution |

**Hybrid approach** (route first, then cascade within a tier) often works best in practice.

### Confidence-Based Cascade Design

**Core mechanism:** Use the model's token probability distribution entropy as an uncertainty proxy. High entropy → escalate.

**Critical problem:** LLM self-reported confidence is **poorly calibrated** — fluent, authoritative responses can be factually wrong with high token probabilities.

**Better approaches (from research):**

1. **Calibrated confidence scores (Amazon Science, 2024):**
   - Logit-based calibration pipeline reduces calibration error by **46%** vs. uncalibrated scores
   - Calibration error-based sampling for efficient calibration data selection
   - Cost-aware cascading policy achieves improved accuracy at **>2× cost reduction** vs. weighted majority voting

2. **Early abstention:**
   - Train models to signal when a query exceeds capability
   - Research achieved **13% cost reduction** and **5% error rate reduction** with only **4.1% increase in abstention rate**

3. **Retrieval-coupled confidence:**
   - Combine model uncertainty with RAG retrieval quality signals
   - Poor retrieval → escalate regardless of model confidence

4. **Empirically calibrated thresholds:**
   - **Non-negotiable** — must evaluate with your specific workload and labeled domain data
   - Academic benchmarks don't predict behavior on rare/unusual inputs where errors are most costly
   - Evaluate routing quality separately on the **tail distribution**

### Cascade for MindForge Verification Pipeline

**MindForge-specific design:**

```
Entry to review
    │
    ▼
┌──────────────────────────┐
│ TIER 0: Local free model │  ← Ollama/Llama-3 or MLX/Llama-3.2
│ (self-consistency × 3)   │
│ Cost: $0                  │
│ Expected: handles 40-60%  │
└──────────┬───────────────┘
           │ (conflict or low conf)
           ▼
┌──────────────────────────┐
│ TIER 1: Cheap API panel  │  ← GPT-4o-mini + Haiku (OpenRouter) + Flash
│ (3 diverse judges)       │
│ Cost: ~$0.002/call        │
│ Expected: handles 20-30%  │
└──────────┬───────────────┘
           │ (disagreement or low conf)
           ▼
┌──────────────────────────┐
│ TIER 2: Mid-tier judge   │  ← GPT-4o or Claude Sonnet (OpenRouter)
│ (single decisive judge)  │
│ Cost: ~$0.01/call         │
│ Expected: handles 5-10%   │
└──────────┬───────────────┘
           │ (still uncertain)
           ▼
┌──────────────────────────┐
│ TIER 3: Premium + web    │  ← GPT-5 / Claude Opus + DuckDuckGo
│ (cross-reference search) │
│ Cost: ~$0.05/call         │
│ Expected: handles 2-5%    │
└──────────┬───────────────┘
           │ (genuinely ambiguous)
           ▼
┌──────────────────────────┐
│ TIER 4: Human review     │  ← MindForge review queue
│ (manual A/R/E/S)         │
│ Cost: human time          │
│ Expected: <2%             │
└──────────────────────────┘
```

### Confidence Threshold Calibration

**Starting thresholds (tune empirically):**

| Transition | Default Threshold | Rationale |
|------------|-------------------|-----------|
| Tier 0 → 1 | Self-consistency disagreement OR avg conf < 0.8 | If 3 samples disagree, the case isn't trivially easy |
| Tier 1 → 2 | < 2/3 judges agree OR avg conf < 0.7 | Panel should reach quorum; if not, escalate |
| Tier 2 → 3 | Judge conf < 0.75 OR contradicts Tier 1 majority | Mid-tier should be decisive; if not, case is genuinely hard |
| Tier 3 → 4 | Judge conf < 0.7 after web search | If even premium + web search can't resolve, human judgment needed |

**Calibration process:**
1. Run cascade on a labeled validation set (~200-500 entries with known correct answers)
2. Measure accuracy at each tier and escalation rate
3. Adjust thresholds to hit target escalation rates (70/20/7/2/1 distribution)
4. Re-calibrate periodically as models and data distributions change

### Semantic Caching (Cost Multiplier)

**Mechanism:** Embed answered queries → store embedding + response. New query → check cosine similarity → return cached response if above threshold.

**Results from real-world RAG pipelines:**
- **3.4× latency reduction** for near-duplicate queries
- **123× latency reduction** for exact matches
- Combined with routing: **60%+ total LLM cost reduction** in high-repetition workloads

**For MindForge:** Many MMLU questions across subjects are similar. A semantic cache keyed on question embeddings + subject could skip redundant judge calls across probe runs.

---

## 6. Practical Architecture for MindForge

### Current State

MindForge's `AutoReviewer` (in `mindforge/review/auto_reviewer.py`) currently uses:
- **Single judge** (auto-detected: OpenAI → OpenRouter → Ollama → MLX)
- **Cross-verification** (judges chosen and rejected independently, compares verdicts)
- **Web search fallback** (DuckDuckGo when confidence < 0.7)
- **No ensemble** — single judge makes the final call

### Proposed Multi-Model Ensemble Architecture

#### New Class: `EnsembleReviewer`

```
mindforge/review/
├── auto_reviewer.py          # Existing single-judge reviewer (keep as fallback)
├── ensemble_reviewer.py      # NEW: Multi-model ensemble reviewer
├── cascade_router.py         # NEW: Confidence-based cascade routing
├── judge_panel.py            # NEW: Panel management + verdict aggregation
└── confidence_calibrator.py  # NEW: Calibrate judge confidence scores
```

#### Architecture Overview

```
                    ┌─────────────────────────┐
                    │    EnsembleReviewer      │
                    │  (orchestrates pipeline) │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    CascadeRouter        │
                    │  (decides tier per entry)│
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
  │  Tier 0: Local │   │ Tier 1: Cheap  │   │ Tier 2-3: API  │
  │  (Ollama/MLX)  │   │  (Panel of 3)  │   │  (Mid/Premium) │
  │                │   │                 │   │                │
  │ MLXAdapter     │   │ OpenAIAdapter   │   │ OpenAIAdapter  │
  │ OllamaAdapter  │   │ OpenRouterAdapt │   │ OpenRouterAdap │
  │                │   │ OllamaAdapter   │   │                │
  └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   JudgePanel        │
                    │  (aggregate votes)  │
                    │  - majority vote    │
                    │  - weighted vote    │
                    │  - consensus check  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ ConfidenceCalibrator │
                    │ (adjust thresholds)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  AutoReviewer       │
                    │  (web search +      │
                    │   final verdict)    │
                    └─────────────────────┘
```

#### Component Design

##### `JudgePanel` — Panel Management & Verdict Aggregation

```python
from dataclasses import dataclass, field
from typing import Optional
from mindforge.probe.adapters import ModelAdapter, create_adapter

@dataclass
class JudgeVerdict:
    """Single judge's verdict on an entry."""
    judge_model: str
    action: str          # "accept" / "reject" / "edit"
    confidence: float     # 0.0 - 1.0
    reasoning: str
    correct: Optional[bool] = None  # None if not evaluated

@dataclass
class PanelResult:
    """Aggregated result from all judges."""
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    final_action: str = "accept"
    final_confidence: float = 0.5
    agreement: str = "unanimous"  # unanimous / majority / split
    escalated: bool = False       # True if disagreement triggers escalation
    reasoning: str = ""

class JudgePanel:
    """Manages a panel of diverse LLM judges and aggregates verdicts."""

    def __init__(self, judge_configs: list[dict]):
        """
        Args:
            judge_configs: List of dicts with 'model' and optional 'weight'.
                e.g. [{"model": "gpt-4o-mini", "weight": 1.0},
                      {"model": "openrouter/anthropic/claude-3-haiku", "weight": 1.0},
                      {"model": "ollama/llama3.1", "weight": 0.7}]
        """
        self.judges: list[tuple[ModelAdapter, float]] = []
        for config in judge_configs:
            adapter = create_adapter(config["model"])
            weight = config.get("weight", 1.0)
            self.judges.append((adapter, weight))

    def evaluate(self, entry: dict) -> PanelResult:
        """Run all judges on an entry, aggregate verdicts."""
        verdicts = []
        for adapter, weight in self.judges:
            verdict = self._judge_with(adapter, entry)
            verdicts.append(verdict)
        return self._aggregate(verdicts)

    def _judge_with(self, adapter: ModelAdapter, entry: dict) -> JudgeVerdict:
        """Single judge evaluation — reuses AutoReviewer's prompt logic."""
        # Build prompt (same as AutoReviewer._build_judge_prompt)
        # Parse response (same as AutoReviewer._parse_judge_response)
        ...

    def _aggregate(self, verdicts: list[JudgeVerdict]) -> PanelResult:
        """Aggregate verdicts using configured method."""
        ...
```

##### `CascadeRouter` — Tier-Based Escalation

```python
class CascadeRouter:
    """Routes entries through cascade tiers based on confidence."""

    TIERS = {
        0: {"adapter_type": "local",   "expected_coverage": 0.50},
        1: {"adapter_type": "cheap",   "expected_coverage": 0.25},
        2: {"adapter_type": "mid",     "expected_coverage": 0.15},
        3: {"adapter_type": "premium", "expected_coverage": 0.07},
        4: {"adapter_type": "human",  "expected_coverage": 0.03},
    }

    DEFAULT_THRESHOLDS = {
        0: 0.80,  # Tier 0 → 1: self-consistency or conf < 0.80
        1: 0.70,  # Tier 1 → 2: panel disagreement or conf < 0.70
        2: 0.75,  # Tier 2 → 3: mid-judge conf < 0.75
        3: 0.70,  # Tier 3 → 4: premium + web search conf < 0.70
    }

    def route(self, entry: dict, tier_result) -> tuple[int, bool]:
        """Decide whether to escalate to the next tier.

        Returns: (next_tier, should_escalate)
        """
        ...
```

##### `ConfidenceCalibrator` — Calibrate Judge Confidence

```python
class ConfidenceCalibrator:
    """Calibrates raw LLM confidence scores using labeled validation data."""

    def __init__(self):
        self.calibration_map: dict[str, list[tuple[float, bool]]] = {}
        # model_name → [(raw_confidence, was_correct), ...]

    def calibrate(self, model_name: str, raw_confidence: float) -> float:
        """Apply calibration mapping to a raw confidence score."""
        # Uses isotonic regression or Platt scaling on stored data
        ...

    def record(self, model_name: str, raw_confidence: float, was_correct: bool):
        """Record a calibration data point from a labeled example."""
        ...

    def save(self, path: str):
        """Persist calibration data to disk."""
        ...

    def load(self, path: str):
        """Load calibration data from disk."""
        ...
```

#### Integration with Existing MindForge Code

**Key design decisions for MindForge compatibility:**

1. **Reuse existing adapters** — `create_adapter()` already handles OpenAI, OpenRouter, Ollama, MLX, Exo. No new adapter code needed.

2. **Reuse AutoReviewer prompt logic** — The judge prompt (`_build_judge_prompt`) and response parser (`_parse_judge_response`) are already battle-tested (including the brace-matching JSON fix). The ensemble just calls them N times.

3. **Reuse web search** — The DuckDuckGo search + answer extraction in `AutoReviewer._web_search()` works as-is for Tier 3 escalation.

4. **Backward compatible** — `AutoReviewer` stays as the single-judge fallback. `EnsembleReviewer` is opt-in:

```python
# CLI: use ensemble (new)
mindforge review --auto --ensemble --limit 10

# CLI: specify panel (new)
mindforge review --auto --ensemble \
  --panel "gpt-4o-mini,openrouter/anthropic/claude-3-haiku,ollama/llama3.1" \
  --limit 10

# CLI: use cascade (new)
mindforge review --auto --cascade --limit 100

# CLI: existing single-judge (unchanged)
mindforge review --auto --judge-model gpt-4o --limit 10
```

5. **Database schema extension** — Add a `panel_verdicts` table to store multi-judge results:

```sql
CREATE TABLE IF NOT EXISTS panel_verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL,
    judge_model TEXT NOT NULL,
    judge_action TEXT NOT NULL,      -- accept/reject/edit
    judge_confidence REAL NOT NULL,
    judge_reasoning TEXT,
    tier INTEGER DEFAULT 0,
    escalated BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entry_id) REFERENCES training_entries(id)
);
```

#### Parallel Execution

Judges in a panel can run **concurrently** since they're independent:

```python
import concurrent.futures

def evaluate_parallel(self, entry: dict, max_workers: int = 3) -> PanelResult:
    """Run all judges concurrently."""
    verdicts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(self._judge_with, adapter, entry): adapter
            for adapter, _ in self.judges
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                verdict = future.result(timeout=30)
                verdicts.append(verdict)
            except Exception as e:
                logger.warning(f"Judge failed: {e}")
                verdicts.append(JudgeVerdict(
                    judge_model="unknown",
                    action="accept",
                    confidence=0.5,
                    reasoning=f"Judge call failed: {e}",
                ))
    return self._aggregate(verdicts)
```

**Caveat:** MindForge currently adds a 0.5s delay between judge calls (`review_batch`) for API rate limiting. With parallel execution, all judges fire simultaneously — ensure each uses a different API key / provider to avoid hitting a single rate limit.

#### Error Handling & Fallback

```
If a judge in the panel fails:
  1. Log the failure
  2. Continue with remaining judges (don't block the panel)
  3. If < 2 judges succeed → escalate to next tier
  4. Store failure info in panel_verdicts table for diagnostics

If all judges fail:
  1. Fall back to AutoReviewer (single judge)
  2. If single judge also fails → mark entry for human review
```

#### Semantic Caching Layer (Optional Enhancement)

```python
class SemanticCache:
    """Cache judge verdicts by question similarity."""

    def __init__(self, threshold: float = 0.92, db_path: str = None):
        self.threshold = threshold
        # Store: question_embedding → (verdict, model, confidence)
        # On lookup: embed question, find nearest neighbor above threshold

    def get(self, question: str, model: str) -> Optional[JudgeVerdict]:
        """Check if we've already judged a similar question."""
        ...

    def put(self, question: str, verdict: JudgeVerdict):
        """Store a verdict for future reuse."""
        ...
```

**MindForge benefit:** MMLU questions are standardized and reused across probe runs. A semantic cache means the second run of `mindforge probe --model X --subject math` can skip re-judging questions that were already verified in a previous run with a different model.

---

## Summary: Recommended Implementation Path

### Phase 1: JudgePanel (Week 1-2)

- Implement `JudgePanel` class with majority vote + weighted vote
- Add `--ensemble` and `--panel` CLI flags
- Add `panel_verdicts` table
- Test with 3-model panel on a small batch (50 entries)

### Phase 2: CascadeRouter (Week 2-3)

- Implement `CascadeRouter` with default thresholds
- Add `--cascade` CLI flag
- Wire up tier escalation (local → cheap panel → mid → premium → human)
- Test cascade on 200-entry batch, measure tier distribution

### Phase 3: ConfidenceCalibrator (Week 3-4)

- Implement calibration data collection (record raw confidence + correctness)
- Implement isotonic regression calibration
- Tune thresholds based on real escalation data
- Add `--calibrate` CLI command to rebuild calibration on demand

### Phase 4: Semantic Cache + Parallel Execution (Week 4-5)

- Add `SemanticCache` for question-level verdict caching
- Parallelize panel execution with `ThreadPoolExecutor`
- Add caching stats to `mindforge review` output

### Phase 5: FastAPI + Frontend Integration (Week 5-6)

- Add `/api/review/ensemble` endpoint
- Add WebSocket progress events for panel verdicts
- Add ensemble config to Review Dashboard in Xbox Blades UI
- Show per-judge verdicts + agreement visualization

---

## 7. Key References

### Primary Research Papers

1. **Verga et al. (2024)** — "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models" (PoLL)
   - arXiv: [2404.18796](https://arxiv.org/abs/2404.18796)
   - **Key finding:** Panel of smaller diverse models outperforms single GPT-4 judge at 1/7th cost

2. **Hovsepian et al. (2024)** — "Label with Confidence: Effective Confidence Calibration and Ensembles in LLM-Powered Classification"
   - Amazon Science: [link](https://www.amazon.science/publications/label-with-confidence-effective-confidence-calibration-and-ensembles-in-llm-powered-classification)
   - **Key finding:** Calibrated confidence scores + cost-aware cascading = >2× cost reduction vs. weighted voting

3. **Yang et al. (2026)** — "Self-Preference Bias in LLM-as-a-Judge" (NeurIPS 2024)
   - arXiv: [2410.21819](https://arxiv.org/html/2410.21819v1)
   - **Key finding:** Judges are up to 50% more likely to incorrectly pass their own family's failed outputs

4. **Wang et al. (2023)** — "Self-Consistency Improves Chain of Thought Reasoning" (ICLR 2023)
   - arXiv: [2203.11171](https://arxiv.org/abs/2203.11171)
   - **Key finding:** Aggregating reasoning paths provides 10–20 point accuracy lifts

5. **Verga et al. survey (2024)** — "LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods"
   - arXiv: [2412.05579](https://arxiv.org/html/2412.05579v2)
   - Repo: [github.com/CSHaitao/Awesome-LLMs-as-Judges](https://github.com/CSHaitao/Awesome-LLMs-as-Judges)

6. **C3PO (2025)** — "Optimized Large Language Model Cascades"
   - arXiv: [2511.07396](https://arxiv.org/html/2511.07396v1)
   - **Key finding:** Self-supervised framework for optimizing LLM cascades under cost constraints

7. **JudgeBench (ICLR 2025)** — "Benchmark for LLM-Based Judges"
   - arXiv: [2410.12784](https://arxiv.org/abs/2410.12784)
   - 10K-comparison dataset stress-testing judge reliability

8. **Meta-Judge (2025)** — "Leveraging LLMs as Meta-Judges: A Multi-Agent Framework"
   - arXiv: [2504.17087](https://arxiv.org/abs/2504.17087)
   - 3-stage pipeline delivers +15% vs single judge

### Practical Guides & Industry Sources

9. **orq.ai** — "Weak Judges, Strong Panel: An Ensemble Approach to LLM Eval"
   - [orq.ai/blog/llm-juries-in-practice](https://orq.ai/blog/llm-juries-in-practice)
   - Best practical guide on panel composition, disagreement as data, and agreement metrics

10. **Evidently AI** — "LLM-as-a-Judge: A Complete Guide"
    - [evidentlyai.com/llm-guide/llm-as-a-judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
    - Comprehensive guide on judge types, prompt design, and best practices

11. **Arize AI** — "LLM-as-a-Jury: What It Is and How To Implement"
    - [arize.com/llm-as-a-jury/](https://arize.com/llm-as-a-jury/)
    - Implementation guide with research paper summaries

12. **Tian Pan (2025)** — "LLM Routing and Model Cascades"
    - [tianpan.co/blog/2025-11-03-llm-routing-model-cascades](https://tianpan.co/blog/2025-11-03-llm-routing-model-cascades)
    - Best practical guide on routing vs. cascading, semantic caching, and production pitfalls

13. **Adnan Masood** — "Rubric-Based Evaluations & LLM-as-a-Judge"
    - [Medium](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80)
    - Rubric design principles, bias catalog, escalation path from judge disagreement to human adjudication

14. **RouteLLM** — Research-backed routing strategies for cost-performance tradeoffs
    - Maintains 95% of frontier quality while routing 85% to cheaper models (45-85% cost reduction)

### MindForge Internal References

15. **MindForge Skill** — `~/.hermes/skills/mlops/mindforge/SKILL.md`
    - Adapter system: `mindforge/probe/adapters.py` (OpenAI, OpenRouter, Ollama, MLX, Exo)
    - Current reviewer: `mindforge/review/auto_reviewer.py` (single judge + web search)
    - Detection priority: OpenAI → OpenRouter → Ollama → MLX

---

## Appendix A: Quick-Reference Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────┐
│                  ENSEMBLE VERIFICATION CHEAT SHEET              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PANEL SIZE:     3-5 judges (diminishing returns after 5)       │
│  DIVERSITY:      Different FAMILIES, not just different models  │
│  VOTING:         Weighted by confidence > plain majority         │
│  STABILITY:      Direct scoring > pairwise (4x more stable)     │
│  CASCADE:        Cheap first → escalate on low confidence        │
│  ESCALATION:     Target <5% to human, <15% to premium tier      │
│  CALIBRATION:    Empirically tuned on YOUR data, not benchmarks  │
│  CACHING:        Semantic cache = 60%+ cost reduction           │
│                                                                 │
│  KEY NUMBERS:                                                   │
│  • PoLL: 7x cheaper than single GPT-4 judge                     │
│  • RouteLLM: 85% queries to cheap models, 95% quality kept     │
│  • Self-consistency: +10-20 points from same model × 3           │
│  • Cascade + calibration: 2x cost reduction vs weighted voting  │
│  • Comparative verdicts flip 35% on prompt tweaks (use direct!) │
│  • Self-preference: up to 50% false-pass on own family outputs   │
│                                                                 │
│  WHEN TO ESCALATE:                                              │
│  • Any judge confidence < 0.7                                    │
│  • < 2/3 panel agreement                                        │
│  • High-confidence disagreement (both > 0.85, different verdicts)│
│  • Domain outside any judge's specialty                         │
│  • Web search contradicts judge verdict                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
