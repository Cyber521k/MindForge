# Frontier Model Intelligence Levels and Domain Coverage (Mid-2026)

## 1. Frontier LLM Models

| Model | Provider | Key Strengths | Weaknesses | Context |
|-------|----------|---------------|------------|---------|
| Claude Opus 4.6/4.8 | Anthropic | Coding (SWE-bench 80.8%), agentic workflows, nuanced writing | Most expensive, no native audio/video | 1M |
| GPT-5.4/5.5 | OpenAI | Broadest capability, computer use (OSWorld 75%), 128K max output | Cost, no native multimodal | 1.05M |
| Gemini 3.1 Pro/3.5 Flash | Google | Multimodal (audio/video), abstract reasoning (ARC-AGI-2 77.1%), best value | Weaker at sustained coding | 1M |
| Grok 4 | xAI | HLE leader (50.7%), 2M context | Smaller ecosystem | 2M |
| Llama 4 Maverick | Meta | Open-weight, 10M context (Scout), self-hostable | Below closed-source frontier | 10M |
| DeepSeek R1/V3.2 | DeepSeek | Best value, strong math | Weaker coding | 128K |
| Qwen 3.5 (397B) | Alibaba | 17B active MoE, 201 languages, GPQA 88.4% | Limited multimodal | 128K |
| MiniMax M2.5 | MiniMax | SWE-bench 80.2% (matches Claude) | New, less tested | 245K |
| GLM-5/5.1 | Zhipu | SWE-bench 77.8%, 94% of Opus coding | Limited English ecosystem | 128K |
| Kimi K2.5 | Moonshot | HLE 51.8% with tools, 100 agents | Limited availability | 256K |

## 2. Saturated Benchmarks

| Benchmark | Top Score | Status |
|-----------|----------|--------|
| MMLU (57 subjects) | ~93% | Saturated - table stakes |
| GSM8K (grade-school math) | ~99% | Saturated |
| HumanEval (Python coding) | ~93% | Saturated + contaminated |
| HellaSwag (commonsense) | 95%+ | Saturated |
| ARC-Challenge | 96%+ | Saturated |

## 3. Benchmarks That Still Matter

| Benchmark | Category | Top Score | Headroom |
|-----------|----------|-----------|----------|
| SWE-bench Verified | Coding | Claude 80.8% | 19% |
| LiveCodeBench | Coding | Qwen 83.6% | 16% |
| Humanity's Last Exam (HLE) | Reasoning | Grok 50.7% | 49% |
| GPQA Diamond | Science | Gemini 94.3% | 6% |
| Terminal-Bench 2.0 / GAIA | Agentic | GPT-5.3 77.3% | 23% |
| ARC-AGI-2 | Abstract Reasoning | Gemini 77.1% | 23% |
| RULER + BFCL v4 | Context/Tools | 50-65% effective context | 35-50% |

## 4. Domain Coverage Gaps

- Advanced Reasoning: causal, analogical, abductive, inductive
- Agentic Capabilities: tool use, planning, multi-step problem solving
- Code Engineering: system design, debugging, code review, CI/CD
- AI/ML Knowledge: model architecture, training, evaluation, deployment
- Multimodal Understanding: image, audio, video, cross-modal
- Long-Context Retrieval: needle-in-haystack, multi-document synthesis
- Creative Tasks: storytelling, poetry, dialogue
- Real-World Knowledge: current events, pop culture, sports
- Mathematical Reasoning: competition math, proofs, applied math
- Scientific Reasoning: experimental design, hypothesis testing
- Natural Language Understanding: pragmatics, discourse, sentiment
- Instruction Following: multi-constraint tasks, format adherence

## 5. Emerging Evaluation Paradigms

- Dynamic Benchmarks: LiveCodeBench (monthly rolling), SEAL
- Human-Preference Evaluation: Chatbot Arena Elo, LMSYS
- Task-Completion Evaluation: Terminal-Bench, GAIA, SWE-bench
- Multi-Turn Conversation Eval: MT-Bench, SimChat
- Agentic Evaluation: multi-step planning, tool orchestration
- Conformal Prediction: calibrated uncertainty estimates

## 6. Benchmark-to-Real-World Correlation

- Weak correlation between benchmark scores and real-world performance
- Scaffold dependence: SWE-bench scores vary 5-15% based on harness
- Contamination: training data may include benchmark questions
- METR task time horizon: models complete ~8 min autonomous tasks (doubling every ~7 months)
- Gap: benchmarks show 90%+ but real-world agentic tasks show 50-77%
- Key takeaway: MindForge should add task-completion and agentic evaluation

## Sources

- Tech Jacks Solutions, Iternal AI, Stob.AI, DataVLab, FullAI, Epoch AI, Center for AI Safety, ARC Prize, METR
