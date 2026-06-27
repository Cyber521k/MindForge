# LLM Confidence Calibration & Uncertainty Quantification for Automated Review Systems

> **Purpose:** Comprehensive reference for building calibrated, uncertainty-aware automated review pipelines using LLMs. Covers why confidence scores are unreliable, how to fix them, and how to set decision thresholds for accept/reject/edit workflows.

---

## Table of Contents

1. [Why LLM Confidence Scores Are Poorly Calibrated](#1-why-llm-confidence-scores-are-poorly-calibrated)
2. [Calibration Methods](#2-calibration-methods)
3. [Uncertainty Quantification](#3-uncertainty-quantification)
4. [Ensemble Approaches](#4-ensemble-approaches)
5. [Practical Implementation: Calibrating a Judge LLM](#5-practical-implementation-calibrating-a-judge-llm)
6. [Threshold Tuning for Decision Routing](#6-threshold-tuning-for-decision-routing)
7. [Key Metrics Reference](#7-key-metrics-reference)
8. [Source Index](#8-source-index)

---

## 1. Why LLM Confidence Scores Are Poorly Calibrated

### 1.1 The Calibration Problem

A perfectly calibrated model satisfies:

```
P(correct | confidence = p) = p   for all p in [0, 1]
```

If a model says "90% confident" across 1,000 predictions, it should be right ~900 times. LLMs systematically violate this. Most instruction-tuned LLMs plot **below the diagonal** on reliability diagrams — they are overconfident.

### 1.2 Overconfidence Patterns

**Pervasive across model tiers:**

| Model Tier | ECE (Expected Calibration Error) | Notes |
|---|---|---|
| Top-tier (Qwen3-235B, DeepSeek-R1) | 11–16% | Even the best models show significant miscalibration |
| Mid-tier (GPT-4.1, Claude Sonnet 4) | 18–27% | Substantial overconfidence |
| Weak (GPT-4o, GPT-4.1-nano) | 39–74% | Extreme miscalibration |
| Frontier models on hard factoid benchmarks | >70% ECE | Near-total calibration failure on difficult tasks |
| Verbalized confidence ("I'm 85% sure") | >37.7% ECE avg | Nearly useless as an engineering signal without additional calibration |

**Key finding (GPT-4o-mini as classifier):** 66.7% of errors occurred at >80% confidence — the canonical overconfidence pattern where wrong answers carry high confidence.

### 1.3 Root Causes

#### RLHF Degrades Calibration
- **SFT (Supervised Fine-Tuning):** ECE ~0.034 — naturally better calibrated because it optimizes for matching the distribution of correct answers
- **RL-trained variant:** ECE ~0.135 (~4x worse) — reward models in PPO training are biased toward rating high-confidence responses as better because human raters perceive confidence as competence
- **Baseline (no fine-tuning):** ECE ~0.163
- **Rule of thumb:** Assume RLHF fine-tuned models are overconfident by default

The models learn that expressing certainty is rewarded. SFT produces better-calibrated outputs because it optimizes for matching correct answer distributions, not persuading a reward model.

#### Token-Level vs. Semantic Confidence Mismatch
LLMs express confidence in two ways that frequently disagree:
1. **Token logit probabilities:** Raw softmax distribution over vocabulary. Available in open-weight models, mostly absent in commercial APIs.
2. **Verbalized confidence:** What the model says when asked to rate certainty. Generated autoregressively like any other text. Research shows this is nearly useless without additional calibration.

#### Additional Causes
- **Exponentially large output space:** Sequence-level confidence can't be enumerated
- **Semantic equivalence problem:** Semantically equivalent outputs may have very different token-level probabilities (e.g., "Sydney" at 70% vs. "Canberra" at 30% — the model is uncertain but token entropy looks confident)
- **Granularity disagreement:** Generative models exhibit lowest average confidence in the middle of generation, not at start or end
- **API limitations:** Many LLMs only expose top-k token probabilities, breaking classical calibration approaches requiring full logit access

### 1.4 Overconfident Wrong Answers vs. Hallucination

These are architecturally distinct failure modes:

| Failure Mode | Type | Detection |
|---|---|---|
| **Hallucination** | Content problem (fabricated information) | Compare against known facts/corpus |
| **Overconfident wrong answer** | Calibration problem (real thing stated with wrong-high confidence) | Requires calibration metrics — passes format validation and hallucination checks |

In high-stakes domains, these compound. Stanford research found legal hallucination rates of 69–88% combined with systematic overconfidence. Medical studies found LLMs repeat planted false clinical info up to 83% of the time with high confidence.

### 1.5 Underconfidence Patterns

Less common but present in specific scenarios:
- **Non-English languages:** Multilingual LLMs show ECE doubling from 7.3% to 18% in non-English languages, with calibration error correlating with language distance and inversely with pretraining corpus share
- **Noisy RAG contexts:** RAG models exhibit severe overconfidence under noisy evidence, but can also swing to underconfidence when uncertainty compounds
- **Small/weaker models on edge cases:** GPT-4.1-nano shows TH-Score of -0.07, indicating it is so poorly calibrated it is anti-informative

---

## 2. Calibration Methods

All methods below are **post-hoc recalibration** techniques — they fit a simple function on a held-out validation set to map raw confidence scores to better-calibrated probabilities. They do not modify the model itself.

### 2.1 Platt Scaling

**Mechanism:** Fits a logistic function over uncalibrated scores:

```
p = σ(A·s + B)
```

Where A and B are learned from a held-out validation set with binary correctness labels. Originally developed for SVMs; generalizes to any system producing a scalar confidence score.

**Properties:**
- Two-parameter fit is data-efficient — can produce usable estimates from a smaller calibration set
- Operates over sequence-level or token-level confidence scores in LLM contexts
- Preserves prediction rankings (monotonic transformation)

**Empirical Results (LLM-generated code):** Platt scaling produced better-calibrated outputs than uncalibrated scores (ICSE 2025).

**Extension — Multivariate Platt Scaling (MPS):** Combines sub-clause frequency scores across multiple generated samples. Consistently outperforms single-score baselines (demonstrated in text-to-SQL tasks).

**Limitations:**
- Too coarse for tasks where correctness depends on local edit decisions — a single sigmoid mapping can't capture sample-dependent miscalibration patterns
- Can degrade proper scoring performance for strong models
- Assumes a specific (logistic) relationship between raw scores and accuracy

### 2.2 Isotonic Regression

**Mechanism:** Learns a piecewise-constant, monotonically non-decreasing mapping from uncalibrated scores to calibrated probabilities using the **Pool Adjacent Violators Algorithm (PAVA)**. No assumed shape for the calibration function.

**Properties:**
- Non-parametric — more flexible than Platt scaling when the confidence-accuracy relationship is non-linear
- Empirically outperforms Platt scaling in most settings
- Available in scikit-learn: `CalibratedClassifierCV` with `method='isotonic'`

**Limitations:**
- Requires large validation sets to avoid overfitting (more data-hungry than Platt scaling)
- Computational complexity is O(n²)
- Can overfit on small calibration sets
- Not recommended for data-sparse situations

**When to choose isotonic over Platt:** When you have sufficient calibration data (>1,000 labeled examples) and the confidence-accuracy relationship is non-monotonic or non-logistic.

### 2.3 Temperature Scaling

**Mechanism:** Divides the logit vector by a scalar T before applying softmax:

```
calibrated_logit = logit / T
```

- T > 1: distribution flattens, confidence drops (fixes overconfidence)
- T < 1: distribution sharpens, confidence rises (fixes underconfidence)
- T = 1: no change

**Implementation (3 steps):**
1. Complete model training (standard process)
2. Optimize T on held-out validation set by minimizing Negative Log-Likelihood (NLL)
3. Adjust scores: divide logits by T before applying softmax

**Optimal T range:** Research with BERT-based and instruction-tuned models suggests best T values fall between **1.5 and 3**.

**Properties:**
- Adds one parameter, preserves prediction rankings, cheap to compute (milliseconds)
- "Can almost perfectly restore network calibration. Requires no additional training data, takes a millisecond, and can be implemented in 2 lines of code." — Geoff Pleiss
- Highest-return, lowest-cost technique **if you have logit access**

**Limitations:**
- Requires logit access (unavailable in most commercial APIs — only works with open-weight models)
- T must be recalibrated per task/domain and when models change
- Single T can't handle input-dependent overconfidence (post-RLHF models develop miscalibration that varies across inputs)

**Adaptive Temperature Scaling (ATS):**
- Predicts a **per-token temperature** from token-level hidden features, fit on a supervised fine-tuning dataset
- Improved calibration by **10–50%** without hurting task performance
- For any RLHF-tuned model, ATS is a stronger baseline than standard temperature scaling
- Standard temperature scaling still works well for base models before RLHF when miscalibration is roughly uniform

### 2.4 Beta Calibration

**Mechanism:** β-calibration extends standard (average-case) calibration to be **conditional on groups** of QA pairs. Standard calibration averages over all inputs, masking miscalibration within subgroups.

**Formal definition:**

```
h is β-calibrated for distribution P if:
    E[Y | h(Q,A) = p, β(Q,A)] = p    for all p in [0,1]

where β: Q×A → S is a mapping to a finite set (grouping function)
```

Reduces to standard calibration when β is constant (all pairs in one group).

**Why it matters:** A model can be calibrated on average but poorly calibrated for individual users interested in specific topics/domains. Two user groups each receiving confidence 0.8 may have very different actual accuracy rates.

**Two post-hoc methods:**

**(a) β-binning (BB):**
- Uses uniform-mass-double-dipping (UMD) histogram binning per partition
- Partitions calibration data by β(q,a)=s, fits separate UMD calibrator per partition
- Hyperparameter: minimum points per bin b (default ~50)

**(b) Scaling-β-binning (HS-BB / S-BB):**
- Adds hierarchical logistic regression scaling step before β-binning
- Uses partial pooling: fixed effects (B₀, B₁) + random effects (U_s, V_s) per partition
- Model: `Y_i ~ Bernoulli(logit⁻¹(B₀ + U_{s[i]} + (B₁ + V_{s[i]})·h_i))`
- Random intercepts/slopes allow different calibration profiles per partition
- Partial pooling shares information across partitions → handles small partitions
- Generally performs best

**β Instantiation — Embed-Then-Bin (kd-tree):**
1. Embedding: DistilBERT [CLS] token → ℝ⁷⁶⁸
2. Histogram: kd-tree partitions the embedding space, successively splitting along dimensions at medians
3. Each partition contains semantically similar QA pairs
4. Key hyperparameter: maximum tree depth d (d=0 → standard calibration)

**Distribution-free guarantees:**

```
ε = √(log(2N/bα) / 2(b-1)) + ν
```

- For β-binning with true labels: ν=0
- For LM-proxy labels: ν estimated empirically
- Practical guidance: When ν=0 and N=1000, set b≈300 for ε=0.1

**Results:** 10–40% improvement in β-calibration performance over baselines. Optimal kd-tree depth is never zero (confirming standard calibration is suboptimal).

### 2.5 Method Selection Guide

| Method | Parameters | Data Needed | Logits Required? | Best For | Key Limitation |
|---|---|---|---|---|---|
| **Temperature Scaling** | 1 (T) | Small | Yes (logits) | Quick fixes, base models | Single T can't handle input-dependent miscalibration |
| **Platt Scaling** | 2 (A, B) | Small | No (any scalar score) | Data-sparse scenarios, binary tasks | Assumes logistic relationship |
| **Isotonic Regression** | Non-parametric | Large (>1K) | No (any scalar score) | Non-linear confidence-accuracy relationships | Overfits on small data, O(n²) |
| **Beta Calibration** | Per-group | Medium-large | No (any scalar score) | Group-specific calibration, subgroup fairness | More complex implementation |
| **Adaptive Temp. Scaling** | Per-token | Medium | Yes (hidden states) | RLHF-tuned models | Requires white-box access |

---

## 3. Uncertainty Quantification

### 3.1 Token-Level Entropy

**Mechanism:** Quantifies the model's confidence for individual tokens in the output sequence.

**Token entropy:**

```
H_t = -Σ_{w_t ∈ V} p(w_t | context) · log p(w_t | context)
```

**Maximum probability (inverse confidence):**

```
1 - max_i p(y_t = i | context)
```

**Types of uncertainty:**

| Type | Definition | Estimation |
|---|---|---|
| **Aleatoric** | Inherent data ambiguity (multiple valid answers) | Expected entropy: E_θ[H(p_θ)] |
| **Epistemic** | Model ignorance/lack of training data | Total Uncertainty minus Aleatoric |
| **Fitting** | Overfitting on frequent tokens vs. underfitting on rare ones | Prediction discrepancy & context dependence |

**Applications:**
- Hallucination detection at the token level
- Error flagging with abstention (using explicit `[IDK]` token)
- Model cascades: route easy queries to small models, difficult ones to experts

**Limitations:**
- **Sequence length bias:** Naive aggregates (summed log-probs) penalize longer sequences. Mitigation: length-invariant estimation (UNCERTAINTY-LINE)
- **"Wrong consensus" trap:** LLMs can assign high probability to multiple near-synonymous tokens, making token-level entropy misleading. A model might return "Sydney" (incorrect) with 70% confidence and "Canberra" (correct) with 30% — token entropy looks like uncertainty, but the model is actually confident about the wrong answer
- **API access:** Many commercial LLMs only expose top-k token probabilities
- Token-level entropy AUROC for hallucination detection: ~0.55–0.65 (moderate at best)

### 3.2 Semantic Uncertainty

**Source:** Kuhn, Gal, Farquhar (ICLR 2023 Spotlight, arXiv:2302.09664)

**Core problem:** Measuring uncertainty in natural language is challenging due to "semantic equivalence" — different sentences can mean the same thing. Token-level entropy treats "Paris is the capital of France" and "France's capital is Paris" as completely different sequences.

**Semantic entropy pipeline:**
1. **Sample N responses** using stochastic decoding (temperature > 0), typically N=10
2. **Compute semantic similarity** between all pairs of responses
3. **Cluster by meaning** using an entailment model or NLI classifier
4. **Compute entropy over clusters** (semantic clusters, not token distributions)

**Properties:**
- Unsupervised method — requires no labeled data
- Uses only a single model
- Requires no modifications to off-the-shelf language models
- More predictive of model accuracy on QA datasets than token-level baselines

**Extensions:**

**Kernel Language Entropy (NeurIPS 2024):** Fine-grained uncertainty by using kernel methods to measure semantic similarity more precisely.

**Evidential Semantic Entropy (EVSE, EACL 2026):** Uses Evidence Theory to model two kinds of ignorance — aleatoric and epistemic — using token-level probabilities from multiple sampling.

**Adaptive Conformal Semantic Entropy (ACSE):** Combines semantic entropy with conformal prediction:

**ACSE pipeline:**
1. Response generation: Sample n responses using top-p sampling
2. Semantic embedding: Map responses to vector space using sentence encoder (e.g., SBERT)
3. Soft clustering: Hierarchical Agglomerative Clustering (HAC) with cosine similarity
4. Adaptive inflation: Adjust base uncertainty based on "cluster brittleness" features
5. Conformal calibration: Apply decision rule to accept/abstain based on error tolerance α

**ACSE key results:**

| Dataset | ACSE AUROC | Token Entropy AUROC |
|---|---|---|
| TriviaQA | 0.88 | 0.65 |
| CoQA | 0.87 | 0.60 |
| Natural Questions | 0.84 | 0.62 |
| TruthfulQA | 0.82 | 0.55 |

**ACSE optimal parameters:**
- Optimal sampling: n=10 (elbow point; more samples yield marginal gains)
- Clustering threshold: cosine distance ε=0.35 (lower causes over-fragmentation, higher masks contradictions)
- Strongest predictor: Cluster-membership Entropy, followed by Centroid Distance

### 3.3 Conformal Prediction

**Mechanism:** Provides distribution-free, finite-sample statistical guarantees on coverage. Instead of point predictions, conformal prediction produces prediction sets that contain the true answer with probability ≥ 1-α.

**Key property:** For a user-specified error tolerance α (e.g., 0.10), conformal prediction guarantees:
```
P(true answer ∈ prediction set) ≥ 1 - α
```

**Application to LLMs:**
- **Selective prediction:** Accept predictions with high confidence, abstain on low confidence. Guarantees error rate ≤ α on accepted predictions.
- **Abstention policies:** Learn when to abstain. Static thresholds are suboptimal; context-adaptive policies improve accuracy by up to 3.2%.

**Conformal calibration metrics:**
- **Acceptance rate:** Percentage of predictions accepted (higher is better)
- **Average prediction set size (APS):** Smaller sets indicate higher precision (ACSE: 1.07)
- **Calibration stability (SSCV):** Lower is better (ACSE: 0.030)

**CAP (Conformalized Abstention Policies):** Context-adaptive risk control that replaces static thresholds with learned policies. Provides target marginal coverage with finite-sample guarantees.

**LLM-specific conformal methods:** NeurIPS 2024 work developed conformal inference methods for obtaining validity guarantees on LLM outputs, adapting the framework to the generative setting where the output space is exponentially large.

**Limitations:**
- Relies on exchangeability assumption (may not hold for time-varying data)
- Static thresholds can be suboptimal compared to adaptive policies
- Marginal guarantees (not conditional on specific inputs)

---

## 4. Ensemble Approaches

### 4.1 Why Ensembles Help

Different LLMs generalize better in distinct regions of input space due to differences in training corpora, objectives, and architectures. Disagreement among LLM predictive distributions signals **epistemic uncertainty**, while consensus indicates more reliable generalization.

**Key insight:** LLMs make complementary predictions due to differences in training and the Zipfian nature of language. Aggregating outputs leads to more reliable uncertainty estimates.

### 4.2 Simple Ensemble Methods

| Method | How It Works | Pros | Cons |
|---|---|---|---|
| **Majority Voting** | Most common answer wins; ties broken by highest confidence | Simple, robust | Discards rich information in disagreement structure |
| **Confidence-Weighted Voting** | Weight votes by model confidence | Accounts for model certainty | Confidence may be miscalibrated |
| **Entropy-Weighted Voting** | Weight by inverse entropy | Penalizes uncertain models | Same calibration issue |
| **Square-Root Confidence-Weighted** | Weight by √confidence | Dampens extreme confidence | Heuristic; no theoretical basis |

**Ensemble calibration improvement:** Up to **46% reduction in calibration error** compared to single-model baselines.

### 4.3 MUSE: Multi-LLM Uncertainty via Subset Ensembles

**Source:** Kruse et al. (EMNLP 2025)

**Core idea:** Not all models contribute equally. MUSE uses **Jensen-Shannon Divergence (JSD)** to identify and aggregate well-calibrated subsets of LLMs.

**Uncertainty decomposition:**

```
Epistemic uncertainty (inter-model disagreement):
    U_epis(S) = (1/|S|) Σ_{i∈S} JS(p_i ‖ p̄)

Aleatoric uncertainty (intrinsic noise):
    U_alea(S) = (1/|S|) Σ_{i∈S} H(p_i)

Total uncertainty:
    U(S) = U_epis(S) + β · U_alea(S)
```

**Two selection strategies:**

| Strategy | Approach |
|---|---|
| **Greedy** | Starts with most confident prediction; iteratively adds models that increase epistemic diversity up to tolerance ε_tol |
| **Conservative** | Only adds models when inclusion meaningfully reduces total uncertainty; more selective and stable |

**Key results (TruthfulQA):**
- Single best model (DS-Qwen): AUROC 72.89, ECE 57.30 (poor calibration despite good discrimination)
- MUSE Conservative (2-model subset): AUROC 72.33, ECE 38.15 (comparable discrimination, substantially better calibration)

**Aggregation strategies:**
1. Simple unweighted mean
2. Aleatoric-aware weighting: predictions weighted by `1 - H(p̂_yes_i)` (higher weight to more confident/low-entropy predictions)

### 4.4 DiscoUQ: Structured Disagreement Analysis

**Source:** Jiang (arXiv:2603.20975, 2026)

**Core problem:** Vote counting treats all disagreements identically. A 3-to-2 split where the minority has weak arguments vs. one where the minority introduces compelling new evidence both get 60% confidence. This discards rich semantic information.

**Framework:** K=5 role-specialized agents (Analytical Reasoner, Devil's Advocate, Knowledge-Focused, Intuitive Responder, Systematic Verifier) each independently produces reasoning and an answer.

**Two feature families:**

**1. Linguistic Structure Features (9 total):**

| Feature | Description |
|---|---|
| Evidence overlap | Shared facts/evidence between majority & minority |
| Minority new information | Genuinely new arguments from minority |
| Minority argument strength | Logical soundness of minority reasoning |
| Majority confidence language | Assertive vs. hedging language |
| Reasoning complexity | Overall complexity |
| Divergence depth | Stage where reasoning diverges (early/middle/late) |

**2. Embedding Geometry Features (8 total):**
- Overall dispersion (mean pairwise cosine distance)
- Majority/minority cohesion
- Cluster distance (majority vs. minority centroids)
- Minority outlier degree
- PCA variance ratio

**Results (average across 4 benchmarks):**

| Method | Avg AUROC | Avg ECE |
|---|---|---|
| Vote Count (baseline) | 0.770 | 0.084 |
| Verbalized Confidence (baseline) | 0.724 | 0.128 |
| LLM Aggregator (baseline) | 0.791 | 0.098 |
| **DiscoUQ-LLM** | **0.802** | **0.036** |
| DiscoUQ-Embed | 0.773 | 0.040 |
| DiscoUQ-Learn (MLP) | 0.790 | 0.042 |

**Key finding:** DiscoUQ-LLM achieves the best ECE (0.036) — 2.3x better calibration than vote counting — while also improving discrimination. The linguistic structure features are most interpretable and offer the best cost-performance ratio.

### 4.5 LLM-as-a-Fuser

**Source:** arXiv:2508.06225

Transforms LLMs from passive judges to active fusers by synthesizing decisions **and critiques** from multiple models, enabling evidence-aware aggregation.

**Results on JudgeBench:**

| Method | Accuracy | ECE |
|---|---|---|
| Entropy-Weighted Voting | 81.71% | 8.48% |
| Confidence-Weighted Voting | 80.00% | 10.43% |
| Majority Voting | 80.00% | 10.77% |
| **Fuser (Qwen3-235B)** | **86.29%** | **6.42%** |

The Fuser framework improves both accuracy and calibration by having a strong model read all model responses + critiques and produce a fused judgment.

### 4.6 Ensemble Trade-offs

| Aspect | Benefits | Challenges |
|---|---|---|
| Performance | 46% reduction in calibration error | Higher computational requirements |
| Scalability | Easy to parallelize | Requires more infrastructure |
| Flexibility | Works across domains | Model compatibility issues |
| Maintenance | Improves reliability | More complex update processes |
| Cost | Better decisions | N× inference cost for N models |

---

## 5. Practical Implementation: Calibrating a Judge LLM

### 5.1 Overview

Calibrating an LLM judge means ensuring that when it scores an output as "8/10 correct" or "confidence: 0.85," that score matches the true probability of correctness as measured against ground truth. Without calibration, a score of 8/10 is an uninterpretable number.

### 5.2 Data Requirements

**What you need:**

| Data Type | Purpose | Minimum Size |
|---|---|---|
| **Calibration set** | Input-output pairs with known correct answers (ground truth labels) | 500–1,000 labeled examples |
| **Judge outputs** | The LLM judge's scores/confidence on the calibration set | One per calibration example |
| **Correctness labels** | Binary (correct/incorrect) or continuous (0–1) ground truth | One per calibration example |
| **Human corrections** | Human reviewer agreement/disagreement with judge scores | Sample of judge outputs (≥100) |
| **Validation set** | Separate from calibration set to avoid overfitting | Same size as calibration set ideally |

**For β-calibration (group-specific):** Additional metadata to define groups (topic, domain, difficulty, user segment). Needs ≥50 examples per group for stable estimates.

**For conformal prediction:** Exchangeable calibration data with confidence scores and correctness labels. Guarantee quality depends on calibration set size.

### 5.3 Step-by-Step Calibration Pipeline

#### Step 1: Build the Calibration Dataset
```
1. Collect production traces (input, model output, judge score, judge confidence)
2. Label each with ground truth correctness (human annotation or reference answer)
3. Split: 60% calibration, 20% validation, 20% test
4. Ensure calibration set covers diverse topics, difficulty levels, and output types
```

#### Step 2: Measure Baseline Calibration
```
1. Run judge on calibration set, collect confidence scores
2. Compute ECE (Expected Calibration Error) across 10–15 bins
3. Plot reliability diagram (confidence vs. observed accuracy)
4. Compute Brier score and overconfidence rate
5. Identify miscalibration pattern (overconfident? underconfident? both in different ranges?)
```

#### Step 3: Apply Post-Hoc Calibration
```
Choose method based on constraints:

IF logit access available:
    → Temperature scaling (fastest, simplest)
    → Adaptive temperature scaling if RLHF-tuned model

IF only scalar scores available:
    → Platt scaling if small dataset (<500)
    → Isotonic regression if large dataset (>1000)
    → Beta calibration if subgroup calibration needed

IF no calibration data with true labels:
    → Use LM-proxy labels (weaker guarantee, ν > 0)
    → BaseCal: re-map post-trained hidden states to base model space (>40% ECE reduction)
```

#### Step 4: Validate
```
1. Apply calibrator to validation set
2. Recompute ECE, Brier score, reliability diagram
3. Check for overfitting (validation ECE should be close to calibration ECE)
4. If using β-calibration: check per-group calibration error
5. Report: "Calibrated confidence of 0.85 means the judge is correct ~85% of the time"
```

#### Step 5: Deploy and Monitor
```
1. Deploy calibrator as a post-processing layer (no model changes)
2. Monitor ECE on rolling window of production data with ground truth
3. Set up drift detection: alert when ECE exceeds threshold (e.g., >0.10)
4. Schedule recalibration cadence (see below)
```

### 5.4 Maintaining Calibration Over Time

Calibration is not a one-time operation. It degrades due to:

| Degrader | Cause | Detection | Fix |
|---|---|---|---|
| **Model updates** | New model version changes confidence distribution | ECE spike after deployment | Recalibrate on new calibration set |
| **Distribution shift** | Input distribution changes over time | Drift in input features/topics | β-calibration with topic-aware partitioning |
| **Prompt changes** | Judge prompt modifications alter scoring behavior | ECE change after prompt edit | Recalibrate with new prompt |
| **Score drift** | Inconsistent scoring of same answer over time | Rolling ECE monitoring | Periodic recalibration (weekly/monthly) |
| **Adversarial adaptation** | Users/game theory adaptation | Anomaly detection on input patterns | Adversarial testing + recalibration |

**Recalibration cadence:**
- **High-stakes (medical, legal, safety):** Weekly recalibration, daily ECE monitoring
- **Standard production:** Monthly recalibration, weekly ECE monitoring
- **Low-stakes:** Quarterly recalibration, monthly ECE monitoring

**Drift detection rule:** Alert when rolling 7-day ECE exceeds `baseline_ECE + 2σ`.

### 5.5 The Alignment Workflow (Human-in-the-Loop)

Based on LangChain's calibration methodology:

```
1. COLLECT: Reviewers examine LLM judge scores and correct disagreements
2. BUILD: Use corrected traces as few-shot examples in the judge prompt
3. MEASURE: Track agreement rate between LLM and human experts
4. ITERATE: Repeat until agreement reaches target (typically ~80%)
```

**Key principle:** Prompt iteration alone is insufficient for reliability. True alignment requires a measurable loop of human corrections and few-shot calibration.

### 5.6 Known Judge Biases and Mitigations

| Bias | Description | Mitigation |
|---|---|---|
| **Position bias** | Favoring first or last option in pairwise comparison | Randomize order; run both permutations |
| **Verbosity bias** | Higher scores for longer answers | Penalize unnecessary length; separate style from correctness |
| **Self-preference** | Favoring outputs from same model family | Use cross-model judging or ensemble methods |
| **Instruction-following** | Rewarding politeness/format over actual goal achievement | Use specific rubrics prioritizing logic over tone |
| **Score drift** | Inconsistent scoring over time | Periodic recalibration + golden set monitoring |
| **False confidence** | Numerical precision masks underlying inaccuracy | Calibrate against ground truth; use ECE monitoring |

### 5.7 Hybrid Evaluation Stack

Don't use LLMs for everything. Use a tiered approach:

```
Layer 1: Deterministic Rules
    → Format, length, regex checks (fast, cheap)
    → Pass/fail binary decisions

Layer 2: Traditional Metrics
    → Exact match, semantic similarity against ground truth
    → BLEU/ROUGE for reference comparison

Layer 3: LLM Judges (calibrated)
    → Nuance, tone, helpfulness, instruction following
    → Use pairwise comparison (more reliable than absolute scoring)

Layer 4: Human Annotation Queue
    → High-stakes domains (medical/legal)
    → Low-confidence cases flagged by calibration layer
    → Creating "golden" datasets for ongoing calibration
```

---

## 6. Threshold Tuning for Decision Routing

### 6.1 The Decision Framework

For an automated review system, calibrated confidence drives a three-way decision:

```
                    ┌─────────────────────┐
                    │  Calibrated Conf.   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         ≥ θ_accept      θ_edit ≤ c < θ_accept    < θ_edit
              │                │                │
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────────┐   ┌──────────┐
        │  ACCEPT  │   │ EDIT/REVIEW  │   │  REJECT  │
        │(auto-    │   │(human-in-    │   │(human    │
        │ approve) │   │ the-loop or  │   │ review   │
        │          │   │ revision)    │   │ required)│
        └──────────┘   └──────────────┘   └──────────┘
```

### 6.2 Setting Thresholds

#### Accept Threshold (θ_accept)
The confidence above which outputs are automatically accepted without human review.

**How to set it:**
1. Define target accuracy for auto-accepted items (e.g., 95%)
2. On calibration set, find the confidence threshold c* where:
   ```
   P(correct | confidence ≥ c*) ≥ target_accuracy
   ```
3. Set θ_accept = c*
4. Verify coverage: what percentage of outputs are auto-accepted?

**Trade-off:** Higher θ_accept → higher accuracy of accepted items but lower automation rate (more items sent to human review).

#### Reject Threshold (θ_edit)
The confidence below which outputs are rejected/escalated.

**How to set it:**
1. Define maximum acceptable error rate for auto-rejected items (items where the model is likely wrong)
2. On calibration set, find the confidence threshold below which error rate exceeds tolerance
3. Set θ_edit = that threshold
4. Items between θ_edit and θ_accept enter the edit/review queue

#### Practical Example
```
Target: Auto-accepted items must be correct ≥95% of the time
         Auto-rejected items (likely wrong) should have <30% accuracy

Calibration set analysis:
  Confidence ≥ 0.92: accuracy = 96.2%, coverage = 45% of all outputs
  Confidence 0.70–0.92: accuracy = 78.4%, coverage = 35%
  Confidence < 0.70: accuracy = 28.1%, coverage = 20%

Result:
  θ_accept = 0.92  (auto-accept 45% of outputs at 96.2% accuracy)
  θ_edit = 0.70    (auto-reject/escalate 20% of outputs)
  Middle 35% → human review or automated revision
```

### 6.3 Cost-Aware Threshold Optimization

Instead of pure accuracy targets, optimize for expected cost:

```
Expected cost = P(accept) × C(false_accept)
              + P(reject) × C(human_review)
              + P(edit) × C(human_edit)
```

Where:
- C(false_accept) = cost of a wrong answer being auto-approved (high for safety-critical)
- C(human_review) = cost of human reviewer time (moderate)
- C(human_edit) = cost of human editing/revising (lower than full review)

**Optimal threshold:** Choose θ_accept and θ_edit to minimize expected cost.

**Example cost structure:**
```
C(false_accept) = $100  (downstream damage from wrong answer)
C(human_review) = $5    (reviewer time per item)
C(human_edit)   = $3    (quick fix per item)

If calibrated confidence = 0.85:
  P(correct) = 0.85, P(wrong) = 0.15
  Expected cost of accepting = 0.15 × $100 = $15
  Expected cost of reviewing = $5
  
  → Review is cheaper → set θ_accept above 0.85 for this cost structure

If C(false_accept) = $20:
  Expected cost of accepting = 0.15 × $20 = $3
  → Accept is cheaper → θ_accept can be lower
```

### 6.4 Conformal Thresholds (Distribution-Free Guarantees)

Instead of heuristic thresholds, use conformal prediction for guaranteed error rates:

```
Given error tolerance α (e.g., 0.05):
  Conformal threshold guarantees: P(error | accepted) ≤ α

1. Compute non-conformity scores on calibration set
2. Find the (1-α) quantile q* of non-conformity scores
3. Accept prediction if its non-conformity score ≤ q*
4. Guarantee: error rate on accepted predictions ≤ α (with high probability)
```

**Adaptive conformal abstention:** Learn context-dependent thresholds instead of static ones. Improves accuracy by up to 3.2% over static thresholds while maintaining the same error guarantee.

### 6.5 Threshold Monitoring and Adjustment

```
Monitor:
  - Automation rate (% auto-accepted): target ≥40% for ROI
  - Error rate among auto-accepted: target ≤ target_accuracy
  - Human review load: % sent to review queue
  - Calibration drift: rolling ECE over time

Adjust:
  - If error rate among accepted > target: raise θ_accept
  - If automation rate too low (<20%): lower θ_accept (accept more risk)
  - If human review load too high: invest in better calibration to narrow the review band
  - If calibration drift detected: recalibrate before adjusting thresholds
```

### 6.6 Rubric Design for Confidence Elicitation

When the judge LLM must produce confidence scores:

```text
You are evaluating an AI-generated answer.
Score the answer using the rubric given below.
Do not reward verbosity. Do not reward self-preference. Do not infer missing facts.
If information is incorrect, reduce the correctness score.

Rubric:
- Correctness (0-4): factual accuracy, logical validity
- Relevance (0-3): alignment with the question intent
- Clarity (0-2): readability and structure
- Safety (0-1): absence of hallucinations or harmful claims

Provide:
1. A numeric score for each dimension
2. A confidence score (0-100) reflecting your certainty
3. A short justification for each dimension
```

**Rubric design principles:**
- Use binary or low-precision scales (LLMs struggle with fine-grained numerical distinctions like 1-10)
- Define scoring anchors and failure conditions explicitly
- Use structured output schemas to ensure consistency
- Chain-of-thought prompting before scoring improves correlation with human judgments
- Pairwise comparisons are more reliable than absolute scoring for calibration

---

## 7. Key Metrics Reference

### Calibration Metrics

| Metric | Formula | Interpretation | Target |
|---|---|---|---|
| **ECE** | Σ (|B_m|/n) · |acc(B_m) - conf(B_m)| | Average gap between confidence and accuracy | <0.05 (good), <0.10 (acceptable) |
| **Brier Score** | (1/n) Σ (p_i - y_i)² | Combined calibration + refinement | Lower is better |
| **MCE** | max_m |acc(B_m) - conf(B_m)| | Worst-case calibration error | Context-dependent |
| **NLL** | -(1/n) Σ log p_i | Negative log-likelihood | Lower is better |
| **TH-Score** | (e^(acc-0.5) - 1) × percentage | Focuses on high/low confidence regions | Higher is better |
| **AUROC** | ROC curve area | Discrimination (can it separate correct/wrong?) | >0.80 (good), >0.70 (acceptable) |
| **AUAC** | Accuracy-confidence curve area | Selective prediction quality | Higher is better |

### Uncertainty Metrics

| Metric | What It Measures | Method |
|---|---|---|
| **Token entropy** | Per-token uncertainty | -Σ p(w) log p(w) |
| **Semantic entropy** | Uncertainty over meanings | Cluster responses, compute cluster entropy |
| **JSD** | Inter-model disagreement | Jensen-Shannon Divergence between model distributions |
| **Vote entropy** | Ensemble disagreement | 1 - H(votes)/log₂(K) |
| **Coverage@90** | % of predictions accepted at 90% confidence | Conformal prediction metric |
| **SSCV** | Calibration stability | Conformal prediction stability metric |

### Production ECE Benchmarks

| Context | Typical ECE |
|---|---|
| SFT models | 0.034 |
| RLHF models | 0.135 |
| Baseline (no fine-tuning) | 0.163 |
| Most production LLMs | 0.05–0.20 |
| Frontier models on hard factoid benchmarks | >0.70 |
| Verbalized confidence | >0.377 |
| After temperature scaling | 0.02–0.05 (for base models) |
| After isotonic regression | 0.01–0.04 (with sufficient data) |
| After ensemble + calibration | <0.04 (DiscoUQ-LLM: 0.036) |

---

## 8. Source Index

### Primary Research Papers

1. **Overconfidence in LLM-as-a-Judge** — arXiv:2508.06225v2 (Aug 2025). Identifies overconfidence phenomenon in LLM judges, proposes TH-Score metric and LLM-as-a-Fuser framework. Evaluated 14 models on JudgeBench.

2. **Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG** — Kuhn, Gal, Farquhar. ICLR 2023 Spotlight. arXiv:2302.09664. Introduces semantic entropy for unsupervised uncertainty estimation.

3. **β-Calibration of Language Model Confidence Scores for Generative QA** — Manggalaa et al. arXiv:2410.06615v1 (Oct 2024). Group-conditional calibration with distribution-free guarantees.

4. **DiscoUQ: Structured Disagreement Analysis for UQ in LLM Agent Ensembles** — Jiang. arXiv:2603.20975v1 (Mar 2026). Extracts linguistic and embedding geometry features from multi-agent disagreement.

5. **MUSE: Simple Yet Effective Multi-LLM Uncertainty Quantification** — Kruse et al. EMNLP 2025. Information-theoretic approach using JSD for subset ensemble selection.

6. **ACSE: Adaptive Conformal Semantic Entropy** — Karimi et al. arXiv:2605.04295v1. Combines semantic entropy with conformal prediction for distribution-free guarantees.

7. **Kernel Language Entropy** — NeurIPS 2024. Fine-grained uncertainty using kernel methods on semantic similarity.

8. **Evidential Semantic Entropy (EVSE)** — EACL 2026. Evidence theory for modeling aleatoric and epistemic ignorance.

9. **Conformal Abstention Policies** — arXiv:2502.06884v1. Context-adaptive risk control for LLM abstention.

10. **LLM Validity via Enhanced Conformal Prediction** — NeurIPS 2024. Conformal inference methods for LLM output validity guarantees.

### Practical Guides & Frameworks

11. **LLM Confidence Calibration in Production** — tianpan.co (Apr 2026). Production patterns: abstention thresholds, temperature scaling, activation-based probes.

12. **A Deep Dive into Calibration of Language Models** — KDnuggets. Comparison of Platt scaling, isotonic regression, temperature scaling for LLMs.

13. **5 Methods for Calibrating LLM Confidence Scores** — Latitude.so (Mar 2025). Practical implementation guide with scikit-learn integration.

14. **How to Calibrate LLM-as-Judge with Human Corrections** — LangChain. Alignment workflow: collect corrections → build few-shot → measure agreement.

15. **LLM-as-a-Judge Calibration: Power & Limits** — Deepchecks. Bias mitigation, rubric design, drift analysis, hybrid systems.

16. **What is LLM-as-a-Judge?** — Braintrust. Evaluation patterns, pipeline building, CI/CD integration.

17. **Confidence Calibration in LLMs** — EmergentMind (Jan 2026). Comprehensive survey: elicitation, correction, multilingual, architectural calibration.

18. **scikit-learn Probability Calibration** — scikit-learn docs. Implementation reference for Platt scaling and isotonic regression.

19. **Tuning the Decision Threshold for Class Prediction** — scikit-learn docs. Threshold optimization methods.

### Key Findings Summary

- **LLMs are systematically overconfident**, with ECE ranging from 11% (best models) to 74% (weak models)
- **RLHF makes overconfidence worse** — reward models bias toward confident-sounding responses
- **Verbalized confidence is nearly useless** without post-hoc calibration (ECE >37%)
- **Temperature scaling is the highest-ROI fix** if you have logit access (2 lines of code, milliseconds)
- **Semantic entropy dramatically outperforms token-level entropy** for uncertainty (AUROC 0.88 vs 0.65)
- **Ensemble disagreement is a strong uncertainty signal** — DiscoUQ achieves ECE of 0.036 vs 0.084 for vote counting
- **Conformal prediction provides distribution-free guarantees** on error rates for abstention decisions
- **β-calibration enables group-specific calibration** — necessary when average calibration masks subgroup miscalibration
- **Calibration must be maintained** — it degrades with model updates, distribution shift, and prompt changes
- **Threshold tuning should be cost-aware** — optimize for expected cost, not just accuracy
