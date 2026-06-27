# DPO Training Data: Quality, Filtering, and Verification

> Research compiled June 2026. Sources: arXiv papers, AI2/HuggingFace documentation, EMNLP/ICML/AAAI/NeurIPS proceedings.

---

## Table of Contents

1. [What Makes High-Quality Chosen/Rejected Pairs](#1-what-makes-high-quality-chosenrejected-pairs)
2. [Automated Filtering Methods](#2-automated-filtering-methods)
3. [Common Failure Modes](#3-common-failure-modes)
4. [Quality Metrics](#4-quality-metrics)
5. [Tools and Frameworks](#5-tools-and-frameworks)
6. [Programmatic Detection and Fixing of Low-Quality Pairs](#6-programmatic-detection-and-fixing-of-low-quality-pairs)
7. [Key Papers Reference](#7-key-papers-reference)

---

## 1. What Makes High-Quality Chosen/Rejected Pairs

### 1.1 Chosen Quality Dominates Everything Else

The single most important finding in recent DPO research (Pan et al., "What Matters in Data for DPO?", arXiv 2508.18312, Aug 2025):

- **Chosen response quality is the dominant factor** in DPO performance. Improving chosen quality consistently boosts performance regardless of rejected response quality.
- **Rejected response quality has relatively limited impact.** Varying the rejected response while keeping chosen fixed produces minimal performance change.
- DPO is primarily **quality anchoring** — learning characteristics of high-quality chosen responses — rather than margin maximization.

Experimental evidence (Llama-3.1-8B, UltraFeedback):

| Configuration | AlpacaEval-2 LC | MMLU | IFEval |
|---|---|---|---|
| Best chosen / Worst rejected | **36.5** | **64.8** | **77.4** |
| Best chosen / High rejected | 33.6 | 62.5 | 76.0 |
| Best chosen / Medium rejected | 34.2 | 63.3 | 76.7 |
| Best chosen / Low rejected | 34.5 | 63.4 | 76.9 |
| Low chosen / Worst rejected | 25.8 | 61.4 | 76.5 |
| Medium chosen / Worst rejected | 26.7 | 62.1 | 75.8 |
| High chosen / Worst rejected | 30.9 | 63.7 | 77.0 |

Key takeaway: Fixing chosen quality and varying rejected → small delta. Fixing rejected and improving chosen → monotonic improvement.

### 1.2 Margin Quality

- **Preference gap** (quality difference between chosen and rejected) matters, but is secondary to chosen quality.
- Improving chosen quality (gap held constant): **+7.1 to +8.7** AlpacaEval-2 gain
- Widening preference gap (quality held constant): **+3.0 to +4.6** gain
- Counterfactual gap change only (identical chosen): **+0.8 to +1.4** gain
- Larger gaps help avoid degenerate cases where chosen ≈ rejected, but the primary benefit is encouraging higher-quality chosen responses.

### 1.3 Difficulty Calibration

Research from Qi & Xu (arXiv 2508.04149, "Difficulty-Based Preference Data Selection by DPO Implicit Reward Gap"):

- **Harder examples** (smaller reward gaps between chosen/rejected) produce **larger gradient magnitudes** during DPO optimization.
- The sigmoid weighting factor `σ(-β·Δr)` is maximized at `Δr = 0` (value = 0.5), meaning maximum uncertainty and information content.
- Selecting the hardest 10% of pairs (smallest reward gaps) **outperforms full-dataset training in 88% of DPO cases**.
- Three-stage pipeline: (1) compute implicit reward gaps, (2) sort ascending, (3) select below threshold.

**However** — this finding must be balanced with the "chosen quality dominates" finding. The optimal strategy is:
- Select pairs where chosen quality is high AND the pair is at an appropriate difficulty level
- Avoid pairs that are too easy (model already separates them well — wasted gradient)
- Avoid pairs that are too hard or contradictory (noisy signal)

### 1.4 Semantic Diversity

- Diverse Preference Optimization (DivPO, OpenReview) formalizes diversity as a selection criterion using probability, frequency, and LLM-judge criteria.
- Coverage theorem (Pan et al.): If a high-reward response y_h is NOT in the support of the data distribution, DPO cannot increase its likelihood. **Without sufficient coverage of high-quality responses, DPO cannot promote desirable behaviors regardless of loss minimization.**
- The optimal DPO policy is: `π_DPO(y|x) ∝ (π_w(y|x) / π_l(y|x))^{1/β} · π_ref(y|x)` where π_w and π_l are marginal distributions of chosen and rejected responses.
- Practical implication: Ensure preference data covers the full distribution of desired response types — don't over-sample narrow domains.

### 1.5 On-Policy vs Off-Policy Data

- On-policy data (generated from the SFT model being trained) outperforms off-policy data (from other models).
- On-policy data amplifies existing quality — benefits are pronounced when chosen quality is already high, marginal when chosen quality is low.
- Online DPO (where rejected responses come from current policy) effectively reduces to SFT on chosen responses for small β (Theorem 4.5, Pan et al.).

---

## 2. Automated Filtering Methods

### 2.1 Reward Model Scoring

**External Reward Model Scoring:**
- Use a pre-trained reward model (e.g., Skywork-Reward-Gemma-2-27B-v0.2, Skywork-Reward-Llama-3.1-8B-v0.2) to score all chosen and rejected responses.
- Compute reward margin: `m = r(x, y_chosen) - r(x, y_rejected)`
- Filter pairs with low or negative margins (indicates reward model disagrees with the preference label).

**Preference Difference (PD) Metric** (Lin et al., AAAI 2025):
- `PD_R(i) = |R(x, y_1) - R(x, y_2)|` — absolute reward difference between pair members
- Theoretically grounded: MSE between expected and empirical loss is proportional to Var(p) and inversely proportional to E(p)
- Higher PD → greater annotation consistency (83% accuracy at top 20% PD vs <50% at bottom 20%)
- Optimal filtering ratio: **50-60% of data** on HH-RLHF
- PD-filtered data outperforms full dataset in nearly all cases, even with less data

**Implicit Reward Scoring:**
- DPO's implicit reward: `r_DPO(x,y) = β·log(π_θ(y|x) / π_ref(y|x))`
- Compute implicit reward gap: `Δr_DPO = r_DPO(x, y_w) - r_DPO(x, y_l)`
- Can be computed without an external reward model — uses the policy and reference models directly.

### 2.2 Preference Strength Metrics

**BeeS (Bayesian Aggregation for Preference data Selection)** (Deng et al., arXiv 2502.14560, NeurIPS 2025):
- Fuses external and implicit reward margins into a unified preference probability using Bayesian aggregation.
- Key insight: External and implicit reward margins show notably weak correlation — even max-margin pairs from strong reward models can yield ambiguous preferences under another reward model.
- Formula: `P(y_w > y_l | m¹, m², ..., m^K) = ∏P(m^i) / (∏P(m^i) + ∏(1-P(m^i)))`
- Strict aggregation deprioritizes pairs with low margin from ANY single reward source.
- Using 10% of UltraFeedback achieves 3-8% improvements on AlpacaEval2 across Llama, Mistral, and Qwen.

**AlphaDPO (Adaptive Reward Margin)** (Wu et al., ICML 2025):
- Introduces a dynamic reward margin instead of DPO's fixed margin.
- Addresses two limitations: DPO's dependency on potentially suboptimal reference model, and SimPO's fixed target margin assumption.
- Employs adaptive preference distribution balancing policy and reference models for personalized reward margins.
- Consistently outperforms DPO and SimPO on AlpacaEval 2 and Arena-Hard.

### 2.3 Contradiction Detection

**NLI-Based Contradiction Detection:**
- Use Natural Language Inference (NLI) models to detect when chosen and rejected responses are semantically contradictory vs. simply different quality.
- Flag pairs where NLI predicts "neutral" relationship — the responses aren't directly comparable, making the preference label questionable.

**Sem-DPO (Semantic Consistency Weighting)** (Mohamed et al., arXiv 2507.20133, 2025):
- Adjusts DPO loss with per-sample weight based on semantic distance between winning response and the prompt.
- Weight: `W_α(x, y_w) = exp(-α · d_cos(e(x), e(y_w)))` where e() is a frozen embedding model.
- Pairs with high semantic similarity to the prompt receive higher weights; low similarity pairs are exponentially suppressed.
- Computed offline with no additional training overhead (~10 min increase).
- Achieves 8-12% higher CLIP similarity vs. DPO.

### 2.4 Policy Filtration (PF-PPO, applicable to DPO)

Zhang et al. (arXiv 2409.06957, 2025):
- Reward models are more reliable when giving high or low rewards (clearly good/bad responses) and less reliable in moderate ranges (ambiguous quality).
- **Best-Worst (BW) strategy**: Train on 50% best + 50% worst samples (highest contrast, highest R² = 0.952).
- **Best-Random (BR) strategy**: 50% best + 50% random from rest (R² = 0.922).
- BoN (best-only) performs poorly — contrastive learning needs bad samples too.
- Use R² (coefficient of determination) between rewards and actual scores to predict strategy effectiveness before training.

### 2.5 fDPO (Filtered DPO)

Morimura et al. (EMNLP 2024):
- Discards lower-quality samples that are inferior to what the learning model itself could generate.
- Rationale: If the model can already produce better responses than the "chosen" in a pair, that pair provides no useful signal.
- Implementation: Generate model responses, compare to preference data, filter pairs where model's own output exceeds chosen quality.

---

## 3. Common Failure Modes

### 3.1 Chosen and Rejected Too Similar

**Symptoms:**
- High token overlap between chosen and rejected (e.g., only differ by a few words)
- Low reward margin from external RM
- Low implicit reward gap `Δr_DPO ≈ 0`

**Impact:**
- Weak gradient signal — the sigmoid weighting factor `σ(-β·Δr)` is at 0.5 (maximum uncertainty), but the gradient magnitude `||∂Δr/∂θ||` is small because the responses are nearly identical.
- Model learns almost nothing useful from these pairs.
- If >20% of pairs have >0.9 similarity, the entire dataset is suspect.

**Detection:**
```python
from difflib import SequenceMatcher
similarities = [SequenceMatcher(None, c.lower(), r.lower()).ratio()
                for c, r in zip(chosen, rejected)]
# Flag pairs with similarity > 0.9
# Flag datasets where >20% of pairs exceed 0.9 similarity
```

### 3.2 Rejected Is Actually Better

**Symptoms:**
- External reward model scores rejected higher than chosen
- Negative reward margin: `m = r(x, y_chosen) - r(x, y_rejected) < 0`
- GPT-4 judge disagrees with the preference label

**Impact:**
- Model learns the wrong direction — actively degrades quality.
- Parameter shrinkage: noisy/flipped preferences cause model parameters to shrink toward zero (Deng et al., 2025).

**Detection:**
- Score all pairs with an external reward model; flag any with negative margins.
- Use multiple reward models and flag pairs where they disagree on direction.
- Use LLM-as-judge (GPT-4) to re-evaluate a sample of pairs.

### 3.3 Factual Errors in Chosen

**Symptoms:**
- Chosen response contains factually incorrect information (wrong dates, numbers, names, claims)
- Rejected response is factually correct but stylistically worse
- Model learns to prefer factually wrong answers

**Impact:**
- DPO amplifies whatever is in the chosen response — if chosen contains errors, the model will reproduce them.
- Particularly damaging for math, code, and factual QA tasks.

**Detection:**
- For verifiable domains (math, code): use rule-based verification (execute code, check answers).
- For factual claims: use retrieval-augmented fact-checking or LLM-as-judge with access to search.
- F-DPO (Factuality-Aware DPO, arXiv 2601.03027): augments DPO pairs with binary factuality indicators and synthetic hallucinated variants.

### 3.4 Hallucinated Content in Chosen

**Symptoms:**
- Chosen response contains fabricated information (fake citations, non-existent references, invented facts)
- Chosen is longer and more detailed than rejected (length bias masking hallucination)

**Impact:**
- Model learns to generate confident-sounding but fabricated content.
- Hallucination can be amplified through the preference signal — DPO increases probability of hallucinated patterns.

**Detection:**
- Citation verification: check that all cited sources actually exist.
- Entity validation: verify named entities (people, places, organizations) against knowledge bases.
- Cross-reference with external knowledge bases or search.
- Length-controlled comparison: if chosen is much longer, check whether the extra content is hallucinated.

### 3.5 Length Bias

**Symptoms:**
- Chosen responses systematically longer than rejected (ratio > 1.5x)
- Model learns to produce longer responses regardless of quality

**Detection:**
```python
chosen_lens = [len(c.split()) for c in chosen]
rejected_lens = [len(r.split()) for r in rejected]
ratio = mean(chosen_lens) / max(mean(rejected_lens), 1)
# Warning if ratio > 1.5 or < 0.67
```

### 3.6 DPO Training Makes Model Worse

**Documented failure** (GitHub rasbt/LLMs-from-scratch #394):
- DPO loss decreases, but model performance degrades on both chosen and rejected text.
- Model performs "much worse on rejected text and a little worse on correct text" — loss still decreases because the gap widens, but absolute quality drops.
- This happens when the preference data quality is low or the learning rate is too high.

**Root causes:**
- Learning rate too high (should be ~10-100x smaller than SFT: 2e-4 SFT → 5e-6 DPO)
- Poor quality preference data (chosen not actually better than rejected)
- Beta too low (model drifts too far from reference)
- Insufficient data diversity

### 3.7 Overfitting and Collapse

- Online DPO (generating both chosen and rejected from the model itself) is more likely to overfit and collapse (OpenReview, "What Matters in Data for DPO?").
- Without sufficient coverage of high-quality responses, DPO cannot promote desirable behaviors regardless of loss minimization (Theorem 4.1).

---

## 4. Quality Metrics

### 4.1 Preference Margin

**Definition:** The reward difference between chosen and rejected responses.

- **External margin**: `m_ex = r_ex(x, y_w) - r_ex(x, y_l)` using an external reward model
- **Implicit margin**: `m_im = r_im(x, y_w) - r_im(x, y_l)` where `r_im(x,y) = β·log(π_θ(y|x) / π_ref(y|x))`
- **DPO implicit reward gap**: `Δr_DPO = β·[log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))]`

**Interpretation:**
- Large positive margin → clear preference signal, easy example
- Near-zero margin → ambiguous, high-difficulty example (maximum gradient signal but potentially noisy)
- Negative margin → preference label likely wrong

**During training (TRL DPOTrainer logs):**
- `rewards/margins`: Average implicit reward margin (should grow during training)
- `rewards/accuracies`: Proportion where chosen reward > rejected reward (should approach 1.0)
- `rewards/chosen`: β·log(π_θ(y⁺|x) / π_ref(y⁺|x)) — should increase
- `rewards/rejected`: β·log(π_θ(y⁻|x) / π_ref(y⁻|x)) — should decrease

### 4.2 Answer Diversity

- Measure distribution of response styles, lengths, and content across the dataset.
- Semantic diversity: embed all chosen responses, measure cluster spread.
- Token-level diversity: type-token ratio, n-gram diversity.
- DivPO framework: uses compression-based diversity metrics.
- Low diversity → model overfits to narrow response patterns.

### 4.3 Coverage Breadth

- Coverage theorem (Pan et al.): DPO cannot promote behaviors not represented in the data distribution.
- Measure: What fraction of the target response distribution is covered by the preference data?
- Practical proxy: Topic diversity of prompts, instruction type diversity, difficulty range coverage.
- Ensure preference data spans: reasoning, creative, factual, coding, safety, instruction-following.

### 4.4 Difficulty Distribution

- **DPO Implicit Reward Gap** as difficulty metric (Qi & Xu, 2025):
  - `Δr_DPO ≈ 0`: Hardest examples (maximum gradient signal)
  - `Δr_DPO >> 0`: Easy examples (model already separates well)
  - Select hardest 10% → outperforms full dataset in 88% of DPO cases
- **IFD (Instruction Following Difficulty)**: Measures semantic overlap between prompt and response.
  - "Low-Gap" = hard samples (Z region in BeeS framework).
- **Gradient magnitude**: `||∂L_DPO/∂θ|| = β·σ(-β·Δr_DPO)·||∂Δr/∂θ||`
  - Sigmoid factor maximized at Δr = 0
  - But gradient also depends on `||∂Δr/∂θ||` — pairs that are similar but not identical maximize this

### 4.5 Preference Consistency (PD Metric)

- Preference Difference (PD): `PD = |R(x, y_1) - R(x, y_2)|`
- Higher PD → greater annotation consistency (83% accuracy at top 20% vs <50% at bottom 20%)
- Lower variance in PD across annotators → more reliable preference signal
- Optimal: filter to top 50-60% by PD.

### 4.6 TRL Training Metrics Summary

| Metric | What It Measures | Healthy Trend |
|---|---|---|
| `loss` | DPO cross-entropy loss | Decreasing |
| `rewards/margins` | Implicit reward gap (chosen - rejected) | Increasing |
| `rewards/accuracies` | Fraction where chosen reward > rejected reward | → 1.0 |
| `rewards/chosen` | Implicit reward for chosen | Increasing |
| `rewards/rejected` | Implicit reward for rejected | Decreasing |
| `entropy` | Token prediction entropy | Moderate, not collapsing |
| `mean_token_accuracy` | Top-1 token match for chosen | Increasing |

If margins are not growing → increase beta or adjust learning rate (Philschmid, 2025).

---

## 5. Tools and Frameworks

### 5.1 TRL (Transformer Reinforcement Learning) — HuggingFace

**Repository:** https://github.com/huggingface/trl
**Docs:** https://huggingface.co/docs/trl/en/dpo_trainer

- **DPOTrainer**: Primary DPO training implementation.
- Supports 15+ loss types: `sigmoid` (default), `hinge`, `ipo`, `robust`, `bco_pair`, `sppo_hard`, `aot`, `apo_zero`, `apo_down`, `discopop`, `sft`, `sigmoid_norm` (SimPO), etc.
- Logs all critical metrics: rewards/margins, rewards/accuracies, rewards/chosen, rewards/rejected, entropy, mean_token_accuracy.
- Supports standard and conversational data formats with auto chat template application.
- Compatible with Q-LoRA, DeepSpeed, gradient checkpointing.

**Key hyperparameters:**
- `beta`: Controls alignment strength (range 0.1-0.5). Higher = less divergence from reference.
- `learning_rate`: Must be ~10-100x smaller than SFT (e.g., 5e-6 for DPO vs 2e-4 for SFT).
- `loss_type`: Choose based on data characteristics (e.g., `robust` for noisy labels with `label_smoothing=0.1`).

### 5.2 RewardBench / RewardBench 2 — AI2

**Repository:** https://github.com/allenai/reward-bench
**Leaderboard:** https://huggingface.co/spaces/allenai/reward-bench

- **First evaluation toolkit for reward models**, including DPO-trained implicit reward models.
- RewardBench 2: Multi-skill benchmark with classification tasks, ties handling (20+ completions per prompt).
- Evaluates: Chat, Chat-Hard, Safety, Reasoning categories.
- Scripts: `run_rm.py` (explicit RMs), `run_dpo.py` (DPO implicit RMs), `run_v2.py` (v2 benchmark).
- Also supports generative reward models (LLM-as-judge) via `rewardbench-gen`.

```bash
# Evaluate a DPO model as a reward model
python scripts/run_dpo.py --model=stabilityai/stablelm-zephyr-3b \
  --ref_model=stabilityai/stablelm-3b-4e1t --batch_size=8
```

### 5.3 OpenRLHF

**Repository:** https://github.com/openrlhf/openrlhf

- Scalable RLHF/DPO training framework.
- Powered by vLLM with Auto Tensor Parallelism and Pipeline Parallelism.
- Supports DPO, iterative DPO, PPO, and reward model training.
- Used in difficulty-based preference data selection research (Qi & Xu, 2025).

### 5.4 Filtered DPO (fDPO)

**Repository:** https://github.com/CyberAgentAILab/filtered-dpo
**Paper:** EMNLP 2024

- Implements sample-level filtering: discards pairs where the model can already generate better responses than the "chosen."
- Provides training scripts for 160M and 1.4B parameter models.
- Poetry-based dependency management.

### 5.5 AlphaDPO

**Repository:** https://github.com/junkangwu/alpha-DPO
**Paper:** ICML 2025

- Implements adaptive reward margin for DPO.
- Built on SimPO codebase.
- Dynamically reparameterizes the reference distribution for instance-specific reward margins.

### 5.6 LM Evaluation Harness

**Repository:** https://github.com/EleutherAI/lm-evaluation-harness

- Standard benchmark suite for evaluating LLMs after DPO training.
- Supports: AlpacaEval, MMLU, GSM8K, IFEval, TruthfulQA, and hundreds more.
- Used in Philschmid (2025) DPO guide for post-training evaluation.

### 5.7 AlpacaEval 2.0

- Standard automatic evaluator for LLM alignment quality.
- GPT-4o as judge, reports Win Rate (WR) and Length-Controlled Win Rate (LCWR).
- Most DPO papers report results on AlpacaEval 2.0.

### 5.8 Reward Model Leaderboard Models

Commonly used reward models for scoring preference data:

| Model | Size | Notes |
|---|---|---|
| Skywork-Reward-Gemma-2-27B-v0.2 | 27B | Used in "What Matters in Data for DPO?" |
| Skywork-Reward-Llama-3.1-8B-v0.2 | 8B | Used in BeeS |
| nicolinho/QRM-Llama3.1-8B-v2 | 8B | Multi-dimensional reward model |
| Fairscale/PairRM | Various | Pairwise comparison model |

---

## 6. Programmatic Detection and Fixing of Low-Quality Pairs

### 6.1 Comprehensive Validation Pipeline

```python
import numpy as np
from difflib import SequenceMatcher
from collections import Counter

class DPODataValidator:
    """Comprehensive DPO preference data quality checker."""

    def __init__(self, dataset):
        self.dataset = dataset  # list of {prompt, chosen, rejected}
        self.flags = []

    def validate_all(self):
        """Run all checks and return flagged pairs."""
        self.check_duplicates()
        self.check_length_bias()
        self.check_response_overlap()
        self.check_empty_short()
        self.check_label_consistency()
        return self.flags

    # --- Check 1: Duplicate pairs ---
    def check_duplicates(self):
        """Identify exact or near-duplicate pairs."""
        seen = set()
        for i, pair in enumerate(self.dataset):
            key = (pair['chosen'].strip().lower(),
                   pair['rejected'].strip().lower())
            if key in seen:
                self.flags.append({
                    'index': i, 'issue': 'duplicate',
                    'severity': 'high'
                })
            seen.add(key)

    # --- Check 2: Length bias ---
    def check_length_bias(self):
        """Flag systematic length differences between chosen and rejected."""
        chosen_lens = [len(p['chosen'].split()) for p in self.dataset]
        rejected_lens = [len(p['rejected'].split()) for p in self.dataset]
        ratio = np.mean(chosen_lens) / max(np.mean(rejected_lens), 1)

        if ratio > 1.5 or ratio < 0.67:
            self.flags.append({
                'index': 'dataset', 'issue': 'length_bias',
                'severity': 'high',
                'detail': f'Mean length ratio: {ratio:.2f}'
            })

        # Flag individual pairs with extreme ratios
        for i, (cl, rl) in enumerate(zip(chosen_lens, rejected_lens)):
            pair_ratio = cl / max(rl, 1)
            if pair_ratio > 3.0 or pair_ratio < 0.33:
                self.flags.append({
                    'index': i, 'issue': 'extreme_length_ratio',
                    'severity': 'medium',
                    'detail': f'Ratio: {pair_ratio:.2f}'
                })

    # --- Check 3: Response overlap ---
    def check_response_overlap(self):
        """Flag pairs where chosen and rejected are too similar."""
        for i, pair in enumerate(self.dataset):
            sim = SequenceMatcher(
                None,
                pair['chosen'].lower(),
                pair['rejected'].lower()
            ).ratio()
            if sim > 0.9:
                self.flags.append({
                    'index': i, 'issue': 'high_overlap',
                    'severity': 'high',
                    'detail': f'Similarity: {sim:.3f}'
                })
            elif sim > 0.75:
                self.flags.append({
                    'index': i, 'issue': 'moderate_overlap',
                    'severity': 'medium',
                    'detail': f'Similarity: {sim:.3f}'
                })

    # --- Check 4: Empty or very short responses ---
    def check_empty_short(self):
        """Flag empty or very short responses."""
        for i, pair in enumerate(self.dataset):
            if len(pair['chosen'].strip()) < 10:
                self.flags.append({
                    'index': i, 'issue': 'short_chosen',
                    'severity': 'high'
                })
            if len(pair['rejected'].strip()) < 10:
                self.flags.append({
                    'index': i, 'issue': 'short_rejected',
                    'severity': 'medium'
                })

    # --- Check 5: Label consistency (requires reward model) ---
    def check_label_consistency(self, rm_scores=None):
        """Flag pairs where reward model disagrees with preference label."""
        if rm_scores is None:
            return  # Skip if no reward model scores provided

        for i, (chosen_score, rejected_score) in enumerate(rm_scores):
            margin = chosen_score - rejected_score
            if margin < 0:
                self.flags.append({
                    'index': i, 'issue': 'flipped_preference',
                    'severity': 'critical',
                    'detail': f'RM margin: {margin:.3f} (rejected scored higher)'
                })
            elif margin < 0.1:  # Threshold depends on RM scale
                self.flags.append({
                    'index': i, 'issue': 'weak_preference',
                    'severity': 'medium',
                    'detail': f'RM margin: {margin:.3f}'
                })
```

### 6.2 Reward Model Scoring Pipeline

```python
class PreferenceDataFilter:
    """Filter preference data using reward model scoring."""

    def __init__(self, reward_model, reference_model=None, beta=0.1):
        self.rm = reward_model  # External RM
        self.ref_model = reference_model  # For implicit reward
        self.beta = beta

    def score_pairs(self, dataset):
        """Score all pairs with external reward model."""
        scores = []
        for pair in dataset:
            chosen_score = self.rm.score(pair['prompt'], pair['chosen'])
            rejected_score = self.rm.score(pair['prompt'], pair['rejected'])
            scores.append((chosen_score, rejected_score))
        return scores

    def compute_implicit_reward(self, policy_model, prompt, response):
        """Compute DPO implicit reward."""
        import torch
        with torch.no_grad():
            policy_logp = policy_model.logprob(prompt, response)
            ref_logp = self.ref_model.logprob(prompt, response)
        return self.beta * (policy_logp - ref_logp)

    def filter_by_margin(self, dataset, scores, min_margin=0.1):
        """Remove pairs with reward margin below threshold."""
        filtered = []
        for pair, (cs, rs) in zip(dataset, scores):
            margin = cs - rs
            if margin >= min_margin:
                filtered.append(pair)
        return filtered

    def filter_by_pd(self, dataset, scores, keep_ratio=0.6):
        """Filter by Preference Difference (PD) metric.
        Keep top keep_ratio% of pairs by PD.
        """
        pds = [abs(cs - rs) for cs, rs in scores]
        threshold = np.percentile(pds, (1 - keep_ratio) * 100)
        return [pair for pair, pd in zip(dataset, pds) if pd >= threshold]

    def filter_by_difficulty(self, dataset, policy_model, keep_ratio=0.1):
        """Select hardest pairs (smallest implicit reward gaps).
        Based on Qi & Xu (2025) difficulty-based selection.
        """
        gaps = []
        for pair in dataset:
            r_chosen = self.compute_implicit_reward(
                policy_model, pair['prompt'], pair['chosen'])
            r_rejected = self.compute_implicit_reward(
                policy_model, pair['prompt'], pair['rejected'])
            gaps.append(r_chosen - r_rejected)

        threshold = np.percentile(gaps, keep_ratio * 100)
        return [pair for pair, gap in zip(dataset, gaps) if gap <= threshold]

    def bayesian_aggregate(self, dataset, external_margins, implicit_margins,
                          L=-2, U=None):
        """BeeS: Bayesian aggregation of multiple margin sources.
        Deng et al. (2025).
        """
        if U is None:
            # Dynamic U: samples in [U, max] < 30 or < max - U
            all_m = external_margins + implicit_margins
            U = np.percentile([abs(m) for m in all_m], 90)

        probabilities = []
        for m_ex, m_im in zip(external_margins, implicit_margins):
            # Clip and normalize each margin source
            p_ex = (np.clip(m_ex, L, U) - L) / (U - L)
            p_im = (np.clip(m_im, L, U) - L) / (U - L)

            # Bayesian aggregation (assumes conditional independence)
            prod_pos = p_ex * p_im
            prod_neg = (1 - p_ex) * (1 - p_im)
            prob = prod_pos / (prod_pos + prod_neg)
            probabilities.append(prob)

        # Return sorted by aggregated probability (descending)
        indices = sorted(range(len(dataset)),
                        key=lambda i: probabilities[i], reverse=True)
        return [dataset[i] for i in indices]
```

### 6.3 Multi-Model Cross-Validation

```python
class MultiModelValidator:
    """Use multiple reward models to cross-validate preference labels."""

    def __init__(self, reward_models):
        """reward_models: list of reward model objects."""
        self.models = reward_models

    def cross_validate(self, dataset):
        """Flag pairs where reward models disagree on direction."""
        all_scores = []
        for model in self.models:
            scores = [(model.score(p['prompt'], p['chosen']),
                       model.score(p['prompt'], p['rejected']))
                      for p in dataset]
            all_scores.append(scores)

        flags = []
        for i in range(len(dataset)):
            directions = []
            for model_scores in all_scores:
                margin = model_scores[i][0] - model_scores[i][1]
                directions.append(1 if margin > 0 else -1)

            agreement = sum(directions) / len(directions)
            if agreement < 0.5:  # Majority disagree with label
                flags.append({
                    'index': i,
                    'issue': 'multi_model_disagreement',
                    'severity': 'critical',
                    'agreement': agreement
                })
            elif agreement < 1.0:  # Some disagreement
                flags.append({
                    'index': i,
                    'issue': 'partial_disagreement',
                    'severity': 'high',
                    'agreement': agreement
                })
        return flags
```

### 6.4 LLM-as-Judge Verification

```python
class LLMJudgeValidator:
    """Use GPT-4 or similar as judge to verify preference labels."""

    def __init__(self, judge_client, judge_model="gpt-4o"):
        self.client = judge_client
        self.model = judge_model

    def judge_pair(self, prompt, chosen, rejected):
        """Ask LLM judge which response is better."""
        judge_prompt = f"""You are an impartial judge. Given a prompt and two responses,
        determine which response is better.

        Prompt: {prompt}
        Response A: {chosen}
        Response B: {rejected}

        Which response is better? Answer with only 'A' or 'B'."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()

        # Returns True if judge agrees with preference label (A=chosen)
        return answer.upper().startswith('A')

    def validate_dataset(self, dataset, sample_size=None):
        """Validate a sample of the dataset with LLM judge."""
        import random
        indices = (random.sample(range(len(dataset)), sample_size)
                   if sample_size else range(len(dataset)))

        flags = []
        for i in indices:
            pair = dataset[i]
            agrees = self.judge_pair(
                pair['prompt'], pair['chosen'], pair['rejected'])
            if not agrees:
                flags.append({
                    'index': i,
                    'issue': 'judge_disagreement',
                    'severity': 'high'
                })
        return flags
```

### 6.5 Recommended Filtering Pipeline (End-to-End)

```python
def filter_preference_dataset(dataset, reward_models, policy_model,
                               ref_model, judge_client=None):
    """
    End-to-end DPO data filtering pipeline.

    Steps:
    1. Basic structural validation (duplicates, length, overlap, empty)
    2. Reward model scoring and margin filtering
    3. Multi-model cross-validation
    4. PD-based filtering (keep top 60%)
    5. Difficulty-based selection (optional: keep hardest 10-50%)
    6. LLM-as-judge spot check (sample validation)
    7. BeeS Bayesian aggregation (if multiple margin sources)

    Returns: filtered dataset + quality report
    """

    # Step 1: Structural validation
    validator = DPODataValidator(dataset)
    structural_flags = validator.validate_all()
    structural_remove = {f['index'] for f in structural_flags
                        if isinstance(f['index'], int) and f['severity'] == 'high'}
    dataset = [d for i, d in enumerate(dataset) if i not in structural_remove]

    # Step 2: Reward model scoring
    filterer = PreferenceDataFilter(reward_models[0], ref_model)
    scores = filterer.score_pairs(dataset)

    # Remove flipped preferences (negative margin)
    dataset = filterer.filter_by_margin(dataset, scores, min_margin=0.0)

    # Step 3: PD-based filtering (keep top 60%)
    dataset = filterer.filter_by_pd(dataset, scores, keep_ratio=0.6)

    # Step 4: Multi-model cross-validation
    multi_validator = MultiModelValidator(reward_models)
    multi_flags = multi_validator.cross_validate(dataset)
    multi_remove = {f['index'] for f in multi_flags
                   if f['severity'] == 'critical'}
    dataset = [d for i, d in enumerate(dataset) if i not in multi_remove]

    # Step 5: Optional LLM judge spot check
    if judge_client:
        judge = LLMJudgeValidator(judge_client)
        judge_flags = judge.validate_dataset(dataset, sample_size=min(100, len(dataset)))
        # Report disagreement rate
        disagreement_rate = len(judge_flags) / min(100, len(dataset))
        if disagreement_rate > 0.15:
            print(f"WARNING: {disagreement_rate:.1%} judge disagreement rate")

    return dataset
```

### 6.6 Monitoring During Training

```python
class DPOTrainingMonitor:
    """Monitor DPO training metrics for quality issues."""

    def __init__(self):
        self.history = {
            'rewards/margins': [],
            'rewards/accuracies': [],
            'rewards/chosen': [],
            'rewards/rejected': [],
            'loss': [],
            'entropy': []
        }

    def update(self, metrics):
        """Call after each logging step."""
        for key in self.history:
            if key in metrics:
                self.history[key].append(metrics[key])

    def check_health(self):
        """Check for training pathologies."""
        issues = []

        # Check 1: Margins not growing
        margins = self.history['rewards/margins']
        if len(margins) > 10:
            recent = np.mean(margins[-5:])
            early = np.mean(margins[:5])
            if recent <= early:
                issues.append({
                    'issue': 'margins_not_growing',
                    'recommendation': 'Increase beta or learning rate'
                })

        # Check 2: Accuracy not improving
        accuracies = self.history['rewards/accuracies']
        if len(accuracies) > 10:
            if np.mean(accuracies[-5:]) < 0.6:
                issues.append({
                    'issue': 'low_accuracy',
                    'recommendation': 'Check data quality or increase beta'
                })

        # Check 3: Chosen rewards decreasing (model getting worse)
        chosen_rewards = self.history['rewards/chosen']
        if len(chosen_rewards) > 10:
            if np.mean(chosen_rewards[-5:]) < np.mean(chosen_rewards[:5]):
                issues.append({
                    'issue': 'chosen_reward_decreasing',
                    'severity': 'high',
                    'recommendation': 'Learning rate may be too high; '
                                     'check data quality'
                })

        # Check 4: Entropy collapse
        entropy = self.history['entropy']
        if len(entropy) > 10:
            if np.mean(entropy[-5:]) < 0.3 * np.mean(entropy[:5]):
                issues.append({
                    'issue': 'entropy_collapse',
                    'severity': 'high',
                    'recommendation': 'Model may be collapsing; '
                                     'reduce learning rate or beta'
                })

        return issues
```

---

## 7. Key Papers Reference

| Paper | Year | Venue | Key Contribution |
|---|---|---|---|
| **What Matters in Data for DPO?** (Pan et al.) | 2025 | arXiv | Chosen quality dominates; coverage theorem |
| **Difficulty-Based Preference Data Selection** (Qi & Xu) | 2025 | arXiv | Implicit reward gap as difficulty metric; 10% data beats 100% |
| **Less is More / BeeS** (Deng et al.) | 2025 | NeurIPS | Bayesian aggregation of margins; parameter shrinkage theory |
| **Data with High PD Are Better** (Lin et al.) | 2025 | AAAI | Preference Difference metric; PD filtering |
| **AlphaDPO** (Wu et al.) | 2025 | ICML | Adaptive reward margin |
| **Filtered DPO** (Morimura et al.) | 2024 | EMNLP | Discard samples worse than model's own output |
| **Policy Filtration for RLHF** (Zhang et al.) | 2025 | arXiv | Reward reliability varies by reward region; BW filtering |
| **Sem-DPO** (Mohamed et al.) | 2025 | arXiv | Semantic consistency weighting |
| **F-DPO / Factuality-Aware DPO** | 2026 | arXiv | Factuality indicators in preference data |
| **DPO** (Rafailov et al.) | 2023 | NeurIPS | Original DPO paper |
| **RewardBench** (Lambert et al.) | 2024 | NAACL | First RM evaluation benchmark |
| **RewardBench 2** | 2025 | arXiv | Multi-skill RM benchmark with ties |
| **DivPO** | 2025 | OpenReview | Diverse preference optimization |

---

## Summary: Practical Recommendations

1. **Prioritize chosen quality above all else.** Invest in ensuring chosen responses are genuinely high-quality. This has more impact than any filtering or optimization trick.

2. **Filter systematically, not just randomly.** Use PD filtering (keep top 50-60%), difficulty-based selection (hardest pairs for stronger gradients), and multi-model cross-validation.

3. **Remove flipped preferences immediately.** Any pair where the reward model scores rejected higher than chosen is either mislabeled or ambiguous. Remove or fix these.

4. **Check for structural issues first.** Duplicates, extreme length ratios, high overlap (>0.9 similarity), and empty responses are easy to catch and fix.

5. **Monitor training metrics.** If rewards/margins aren't growing or rewards/accuracies stay below 0.6, stop and investigate data quality before continuing.

6. **Use multiple reward models.** External and implicit margins show weak correlation. BeeS-style Bayesian aggregation provides more robust filtering than any single source.

7. **10% of well-filtered data can beat 100% unfiltered.** Multiple papers confirm this across different datasets and models. Quality > quantity for DPO.

8. **For noisy labels, use `loss_type="robust"` with `label_smoothing=0.1`** in TRL DPOTrainer. This is an unbiased DPO loss designed for noisy preferences.

9. **Learning rate must be ~10-100x smaller than SFT.** Typical: 5e-6 for DPO vs 2e-4 for SFT. Too-high LR causes model degradation even with good data.

10. **Beta controls alignment strength.** Range 0.1-0.5. If margins aren't growing, increase beta. Higher beta = less divergence from reference model.
