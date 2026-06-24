"""ProbeEngine - runs model probing against MMLU questions."""

import os
import sys
import json
import time
import logging

logger = logging.getLogger(__name__)

# Add parent dirs to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mindforge.probe.adapters import create_adapter, extract_answer_letter
from mindforge.probe.question_gen import (
    resolve_subject, load_mmlu_questions, format_mcq_prompt,
    generate_tier2_followups, generate_tier3_edge_cases,
)
from mindforge.score.answer_key import score_answer
from mindforge.score.confidence import compute_confidence, compute_confidence_with_judge
from mindforge.score.judge import LLMJudge
from mindforge.correct.analyzer import analyze_error
from mindforge.correct.corrector import formulate_correction
from mindforge.format.dpo import format_dpo_entry
from mindforge.vault.database import Database

_LETTERS = ["A", "B", "C", "D"]


class ProbeEngine:
    """Engine that probes a model against MMLU questions and generates DPO training data.

    Phase 2 enhancements:
        - Supports multiple probing tiers (1, 2, 3, or 'all')
        - Supports LLM-as-Judge scoring via judge_model parameter
        - Generates Tier 2 follow-up and Tier 3 edge-case questions
    """

    def __init__(self, model_name, subject, tier=1, limit=25, output_dir=None,
                 judge_model=None):
        """Initialize the probe engine with model, subject, and scoring config.

        Args:
            model_name: HuggingFace repo or model ID to probe.
            subject: MMLU subject name (resolved via taxonomy/subjects.yaml).
            tier: Probing depth (1=breadth, 2=depth, 3=edge cases, or 'all').
            limit: Maximum number of questions to probe.
            output_dir: Directory for DPO output (default: data/training-data/dpo/).
            judge_model: Optional LLM-as-Judge model name for enhanced scoring.
        """
        self.model_name = model_name
        self.subject_input = subject
        self.tier = tier
        self.limit = limit
        self.output_dir = output_dir or os.path.join(_project_root, "data", "training-data", "dpo")
        self.judge_model_name = judge_model

        # Resolve subject
        self.mmlu_subject = resolve_subject(subject)
        if not self.mmlu_subject:
            raise ValueError(
                f"Could not resolve subject '{subject}'. "
                "Check taxonomy/subjects.yaml for valid subject names."
            )

        logger.info(f"Subject '{subject}' resolved to MMLU subject '{self.mmlu_subject}'")

        # Create adapter for the probed model
        self.adapter = create_adapter(model_name)
        logger.info(f"Using adapter: {type(self.adapter).__name__}")

        # Create judge (LLM-as-Judge) if a judge model is specified
        self.judge = None
        if judge_model:
            judge_adapter = create_adapter(judge_model)
            self.judge = LLMJudge(model_adapter=judge_adapter, model_name=judge_model)
            logger.info(f"LLM Judge enabled: {judge_model}")

        # Database
        db_path = os.path.join(_project_root, "data", "mindforge.db")
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        except OSError:
            pass  # Database() will handle/log this
        self.db = Database(db_path)

    def run(self):
        """Run the probing flow end-to-end.

        1. Load MMLU questions
        2. Ask the model each question (Tier 1)
        3. Optionally run Tier 2 (depth) and Tier 3 (edge cases)
        4. Score against answer key (and/or LLM judge)
        5. Generate DPO training pairs
        6. Write to train.jsonl

        Returns:
            dict with summary stats
        """
        # Determine which tiers to run
        if self.tier == "all":
            tiers_to_run = [1, 2, 3]
        elif isinstance(self.tier, int):
            tiers_to_run = list(range(1, self.tier + 1))
        else:
            tiers_to_run = [1]

        print(f"\n{'='*60}")
        print(f"  MindForge Probe")
        print(f"{'='*60}")
        print(f"  Model:   {self.model_name}")
        print(f"  Subject: {self.subject_input} -> {self.mmlu_subject}")
        print(f"  Tiers:   {tiers_to_run}")
        print(f"  Limit:   {self.limit} questions")
        if self.judge:
            print(f"  Judge:   {self.judge_model_name}")
        print(f"{'='*60}\n")

        # Step 1: Load questions
        print("Loading MMLU questions...")
        questions = load_mmlu_questions(self.mmlu_subject, limit=self.limit)
        print(f"Loaded {len(questions)} questions.\n")

        if not questions:
            print("ERROR: No questions found for this subject.")
            return {"error": "no_questions", "total": 0}

        # Step 2: Probe each question (Tier 1)
        results = []
        correct_count = 0
        incorrect_count = 0

        for i, q in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] Probing...", end=" ", flush=True)

            # Format the prompt
            prompt = format_mcq_prompt(q["question"], q["choices"], self.mmlu_subject)

            # Ask the model
            try:
                model_response = self.adapter.ask(prompt, max_tokens=256)
            except Exception as e:
                logger.error(f"Model error on question {i+1}: {e}")
                print(f"ERROR: {e}")
                continue

            # Extract answer letter
            model_answer_letter = extract_answer_letter(model_response)
            correct_answer_letter = _LETTERS[q["answer"]]

            # Score with answer key
            is_correct = score_answer(model_answer_letter, correct_answer_letter)

            # Score with LLM judge (if available)
            judge_verdict = None
            if self.judge:
                judge_verdict = self.judge.judge(
                    question=q["question"],
                    model_answer=model_response,
                    correct_answer=f"{correct_answer_letter}) {q['choices'][q['answer']]}",
                )

            # Compute confidence using enhanced scoring
            confidence = compute_confidence_with_judge(
                is_correct_answer_key=is_correct,
                judge_verdict=judge_verdict,
            )

            if is_correct:
                correct_count += 1
                print(f"✓ Correct (model={model_answer_letter})")
            else:
                incorrect_count += 1
                print(f"✗ Incorrect (model={model_answer_letter}, correct={correct_answer_letter})")

            # Store result
            result = {
                "question_idx": i,
                "prompt": prompt,
                "question": q["question"],
                "choices": q["choices"],
                "correct_answer_idx": q["answer"],
                "correct_answer_letter": correct_answer_letter,
                "model_response": model_response,
                "model_answer_letter": model_answer_letter,
                "is_correct": is_correct,
                "confidence": confidence,
                "subject": self.mmlu_subject,
                "model": self.model_name,
                "tier": 1,
            }
            if judge_verdict:
                result["judge_verdict"] = judge_verdict
            results.append(result)

            # Store in database (batch commit for efficiency)
            self.db.store_response(result)

            # Tier 2: Depth probing (follow-ups based on model's answer)
            if 2 in tiers_to_run:
                tier2_questions = generate_tier2_followups(
                    q["question"], model_response, self.mmlu_subject
                )
                for j, followup in enumerate(tier2_questions):
                    try:
                        t2_response = self.adapter.ask(followup, max_tokens=256)
                        t2_result = {
                            "question_idx": i,
                            "prompt": followup,
                            "question": followup,
                            "choices": [],
                            "correct_answer_idx": -1,
                            "correct_answer_letter": None,
                            "model_response": t2_response,
                            "model_answer_letter": None,
                            "is_correct": None,
                            "confidence": 0.5,
                            "subject": self.mmlu_subject,
                            "model": self.model_name,
                            "tier": 2,
                        }
                        results.append(t2_result)
                        self.db.store_response(t2_result)
                    except Exception as e:
                        logger.warning(f"Tier 2 follow-up failed: {e}")

            # Tier 3: Edge cases (adversarial questions)
            if 3 in tiers_to_run:
                # Only generate edge cases once per subject to avoid duplicates
                if i == 0:
                    self._tier3_questions = generate_tier3_edge_cases(self.mmlu_subject)

                if hasattr(self, "_tier3_questions"):
                    for edge_q in self._tier3_questions:
                        try:
                            t3_response = self.adapter.ask(edge_q, max_tokens=256)
                            t3_result = {
                                "question_idx": i,
                                "prompt": edge_q,
                                "question": edge_q,
                                "choices": [],
                                "correct_answer_idx": -1,
                                "correct_answer_letter": None,
                                "model_response": t3_response,
                                "model_answer_letter": None,
                                "is_correct": None,
                                "confidence": 0.5,
                                "subject": self.mmlu_subject,
                                "model": self.model_name,
                                "tier": 3,
                            }
                            results.append(t3_result)
                            self.db.store_response(t3_result)
                        except Exception as e:
                            logger.warning(f"Tier 3 edge case failed: {e}")
                    # Clear so we don't re-run for every question
                    del self._tier3_questions

        # Detect total failure: all questions were skipped due to model errors
        if len(results) == 0 and len(questions) > 0:
            print("\nERROR: All model calls failed. No results to process.")
            self.adapter.close()
            self.db.close()
            return {
                "error": "All model calls failed. Check the model name and ensure mlx-lm is installed.",
                "total": 0,
                "dpo_entries": 0,
            }

        # Commit all responses at once
        self.db.conn.commit()

        # Step 3: Generate DPO training pairs from incorrect answers
        print(f"\n{'='*60}")
        print(f"  Results Summary")
        print(f"{'='*60}")
        print(f"  Total questions:  {len(results)}")
        print(f"  Correct:          {correct_count}")
        print(f"  Incorrect:        {incorrect_count}")
        if results:
            print(f"  Accuracy:         {correct_count/len(results)*100:.1f}%")
        print(f"{'='*60}\n")

        dpo_entries = []
        for r in results:
            if not r.get("is_correct", True) and r.get("model_answer_letter") is not None:
                # Incorrect answer -> DPO pair
                chosen_text = self._format_answer_response(
                    r["correct_answer_letter"],
                    r["choices"][r["correct_answer_idx"]]
                )
                rejected_text = self._format_answer_response(
                    r["model_answer_letter"],
                    r["choices"][ord(r["model_answer_letter"]) - ord("A")]
                )

                dpo_entry = format_dpo_entry(
                    prompt=r["prompt"],
                    chosen=chosen_text,
                    rejected=rejected_text,
                )
                dpo_entries.append(dpo_entry)

                # Store training entry in DB
                self.db.store_training_entry(
                    response_id=r.get("db_id"),
                    prompt=r["prompt"],
                    chosen=chosen_text,
                    rejected=rejected_text,
                    format="dpo",
                    subject=self.mmlu_subject,
                )

        # Also include some correct answers as weak pairs (up to 30% of incorrect)
        weak_count = min(int(len(dpo_entries) * 0.3), correct_count)
        if weak_count > 0:
            correct_results = [r for r in results if r.get("is_correct")]
            for r in correct_results[:weak_count]:
                chosen_text = self._format_answer_response(
                    r["correct_answer_letter"],
                    r["choices"][r["correct_answer_idx"]]
                )
                rejected_text = "I'm not sure about this question."

                dpo_entry = format_dpo_entry(
                    prompt=r["prompt"],
                    chosen=chosen_text,
                    rejected=rejected_text,
                )
                dpo_entries.append(dpo_entry)

        # Step 4: Write DPO output
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "train.jsonl")

        with open(output_path, "w") as f:
            for entry in dpo_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"\nDPO training data written to: {output_path}")
        print(f"  Total DPO entries: {len(dpo_entries)}")
        print(f"  From incorrect answers: {len([e for e in dpo_entries if 'not sure' not in e['rejected'].lower()])}")
        print(f"  From correct (weak) pairs: {len([e for e in dpo_entries if 'not sure' in e['rejected'].lower()])}")

        # If we don't have enough DPO entries, supplement with correct answers
        if len(dpo_entries) < 20:
            print(f"\n  Only {len(dpo_entries)} entries (< 20). Supplementing with correct answers...")
            needed = 20 - len(dpo_entries)
            correct_results = [r for r in results if r.get("is_correct")]
            for r in correct_results[:needed]:
                chosen_text = self._format_answer_response(
                    r["correct_answer_letter"],
                    r["choices"][r["correct_answer_idx"]]
                )
                wrong_idx = [j for j in range(4) if j != r["correct_answer_idx"]][0]
                rejected_text = self._format_answer_response(
                    _LETTERS[wrong_idx],
                    r["choices"][wrong_idx]
                )

                dpo_entry = format_dpo_entry(
                    prompt=r["prompt"],
                    chosen=chosen_text,
                    rejected=rejected_text,
                )
                dpo_entries.append(dpo_entry)

                with open(output_path, "a") as f:
                    f.write(json.dumps(dpo_entry) + "\n")

            print(f"  Total after supplementation: {len(dpo_entries)}")

        # Cleanup
        self.adapter.close()
        self.db.close()

        return {
            "total_questions": len(results),
            "correct": correct_count,
            "incorrect": incorrect_count,
            "accuracy": correct_count / len(results) if results else 0,
            "dpo_entries": len(dpo_entries),
            "output_path": output_path,
        }

    def _format_answer_response(self, letter, choice_text):
        """Format an answer response for DPO chosen/rejected fields."""
        return f"The answer is {letter}) {choice_text}"
