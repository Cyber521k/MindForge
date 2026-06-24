"""Evaluator module - model evaluation and comparison.

Tries lm-eval-harness first, falls back to mlx_lm.evaluate, then to a simple
MMLU accuracy check using the probe engine.
"""

import os
import sys
import json
import time
import logging
import importlib
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Project root for DB path resolution
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _get_db():
    """Get the Database instance for tracking evaluation results."""
    from mindforge.vault.database import Database
    db_path = os.path.join(_project_root, "data", "mindforge.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return Database(db_path)


def _check_lm_eval_available() -> bool:
    """Check if lm-eval-harness (lm_eval) is available."""
    try:
        importlib.import_module("lm_eval")
        return True
    except ImportError:
        return False


def _check_mlx_lm_evaluate_available() -> bool:
    """Check if mlx_lm.evaluate is available."""
    try:
        importlib.import_module("mlx_lm.evaluate")
        return True
    except ImportError:
        return False


def _run_lm_eval(
    model: str,
    tasks: str,
    num_fewshot: int,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation using lm-eval-harness."""
    print(f"  Using lm-eval-harness for evaluation...")

    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.mlx_lm import MLXLM

        # Create the model
        model_obj = MLXLM(pretrained=model, adapter_path=adapter_path)

        # Run evaluation
        results = simple_evaluate(
            model=model_obj,
            tasks=[tasks] if not isinstance(tasks, list) else tasks,
            num_fewshot=num_fewshot,
        )

        # Extract scores
        scores = {}
        if "results" in results:
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        scores[f"{task_name}_{metric}"] = value

        # Get the primary score
        primary_score = 0.0
        if scores:
            # Try to find accuracy metric
            for key in ["acc", "acc_norm", "exact_match"]:
                for score_key, score_val in scores.items():
                    if key in score_key:
                        primary_score = score_val
                        break

        return {
            "score": primary_score,
            "metric": "accuracy",
            "details": json.dumps(scores),
            "backend": "lm_eval",
        }

    except Exception as e:
        logger.warning(f"lm-eval-harness evaluation failed: {e}")
        raise


def _run_mlx_lm_evaluate(
    model: str,
    tasks: str,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation using mlx_lm.evaluate."""
    print(f"  Using mlx_lm.evaluate for evaluation...")

    try:
        import mlx_lm.evaluate as mlx_eval

        # mlx_lm.evaluate may have different interfaces
        # Try the evaluate function
        if hasattr(mlx_eval, "evaluate_model"):
            results = mlx_eval.evaluate_model(
                model=model,
                tasks=[tasks] if isinstance(tasks, str) else tasks,
                adapter_path=adapter_path,
            )
        elif hasattr(mlx_eval, "run_evaluation"):
            results = mlx_eval.run_evaluation(
                model=model,
                tasks=[tasks] if isinstance(tasks, str) else tasks,
                adapter_path=adapter_path,
            )
        else:
            # Try CLI approach
            import subprocess

            cmd = [
                sys.executable, "-m", "mlx_lm.evaluate",
                "--model", model,
                "--tasks", tasks if isinstance(tasks, str) else ",".join(tasks),
            ]
            if adapter_path:
                cmd.extend(["--adapter-path", adapter_path])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode != 0:
                raise RuntimeError(f"mlx_lm.evaluate failed:\n{result.stderr}")

            # Parse output for accuracy
            score = 0.0
            for line in result.stdout.split("\n"):
                if "acc" in line.lower():
                    try:
                        # Try to extract number
                        import re
                        nums = re.findall(r"[\d.]+", line)
                        if nums:
                            score = float(nums[-1])
                    except (ValueError, IndexError):
                        pass

            results = {"score": score, "metric": "accuracy"}

        score = results.get("score", results.get("accuracy", 0.0))

        return {
            "score": score,
            "metric": results.get("metric", "accuracy"),
            "details": json.dumps(results),
            "backend": "mlx_lm_evaluate",
        }

    except Exception as e:
        logger.warning(f"mlx_lm.evaluate evaluation failed: {e}")
        raise


def _run_simple_mmlu_eval(
    model: str,
    tasks: str,
    num_fewshot: int,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a simple MMLU accuracy check using the probe engine.

    This is the fallback when neither lm-eval-harness nor mlx_lm.evaluate
    are available. It uses MindForge's own probe engine to test the model
    on MMLU STEM questions and compute accuracy.
    """
    print(f"  Using simple MMLU probe-based evaluation...")

    from mindforge.probe.adapters import create_adapter, extract_answer_letter
    from mindforge.score.answer_key import score_answer
    from mindforge.probe.question_gen import resolve_subject, load_mmlu_questions, format_mcq_prompt

    # Map task names to MMLU subjects
    task_to_subject = {
        "mmlu_stem": "high_school_mathematics",
        "mmlu_math": "high_school_mathematics",
        "mmlu_physics": "high_school_physics",
        "mmlu_chemistry": "high_school_chemistry",
        "mmlu_biology": "high_school_biology",
        "mmlu_cs": "computer_science",
        "mmlu": "high_school_mathematics",
    }

    subject = task_to_subject.get(tasks, "high_school_mathematics")
    if isinstance(tasks, list) and len(tasks) > 0:
        subject = task_to_subject.get(tasks[0], "high_school_mathematics")

    # Resolve subject
    mmlu_subject = resolve_subject(subject)
    if not mmlu_subject:
        mmlu_subject = subject

    # Load questions
    questions = load_mmlu_questions(mmlu_subject, limit=25)

    if not questions:
        return {
            "score": 0.0,
            "metric": "accuracy",
            "details": json.dumps({"error": "no_questions", "subject": mmlu_subject}),
            "backend": "simple_mmlu",
        }

    # Create adapter for the model
    try:
        adapter = create_adapter(model, adapter_path=adapter_path)
    except Exception as e:
        logger.error(f"Failed to create adapter: {e}")
        return {
            "score": 0.0,
            "metric": "accuracy",
            "details": json.dumps({"error": str(e)}),
            "backend": "simple_mmlu",
        }

    # Run questions
    correct = 0
    total = 0
    _LETTERS = ["A", "B", "C", "D"]

    for i, q in enumerate(questions):
        prompt = format_mcq_prompt(q["question"], q["choices"], mmlu_subject)
        try:
            response = adapter.ask(prompt, max_tokens=256)
            model_letter = extract_answer_letter(response)
            correct_letter = _LETTERS[q["answer"]]

            if score_answer(model_letter, correct_letter):
                correct += 1
            total += 1

            print(f"  [{i+1}/{len(questions)}] {'✓' if score_answer(model_letter, correct_letter) else '✗'}")

        except Exception as e:
            logger.warning(f"Question {i+1} failed: {e}")
            total += 1

    adapter.close()

    accuracy = correct / total if total > 0 else 0.0

    return {
        "score": accuracy,
        "metric": "accuracy",
        "details": json.dumps({
            "correct": correct,
            "total": total,
            "subject": mmlu_subject,
            "task": tasks,
        }),
        "backend": "simple_mmlu",
    }


def evaluate_model(
    model: str,
    tasks: str = "mmlu_stem",
    num_fewshot: int = 5,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a model on specified tasks.

    Tries lm-eval-harness first, falls back to mlx_lm.evaluate,
    then to a simple MMLU accuracy check using the probe engine.

    Args:
        model: Model name or path
        tasks: Evaluation task(s) - string or list of strings
        num_fewshot: Number of few-shot examples
        adapter_path: Path to adapter weights (for fine-tuned models)

    Returns:
        dict with keys:
            - score: Primary evaluation score (accuracy)
            - metric: Metric name (e.g., "accuracy")
            - details: JSON string with detailed results
            - backend: Which evaluation backend was used
            - model: Model name
            - tasks: Task name(s)
            - status: "completed" or "failed"
    """
    print(f"\n{'='*60}")
    print(f"  MindForge Evaluation")
    print(f"{'='*60}")
    print(f"  Model:      {model}")
    print(f"  Tasks:      {tasks}")
    print(f"  Few-shot:   {num_fewshot}")
    if adapter_path:
        print(f"  Adapter:    {adapter_path}")
    print(f"{'='*60}\n")

    # Initialize database tracking
    db = _get_db()

    # Try each evaluation backend in order
    backends_to_try = []

    if _check_lm_eval_available():
        backends_to_try.append(("lm_eval", _run_lm_eval))
    else:
        logger.info("lm-eval-harness not available")

    if _check_mlx_lm_evaluate_available():
        backends_to_try.append(("mlx_lm_evaluate", _run_mlx_lm_evaluate))
    else:
        logger.info("mlx_lm.evaluate not available")

    # Always have the simple fallback
    backends_to_try.append(("simple_mmlu", _run_simple_mmlu_eval))

    results = None
    backend_used = None
    error = None

    for backend_name, backend_fn in backends_to_try:
        try:
            print(f"  Trying {backend_name}...")
            results = backend_fn(
                model=model,
                tasks=tasks,
                num_fewshot=num_fewshot,
                adapter_path=adapter_path,
            )
            backend_used = backend_name
            break
        except Exception as e:
            logger.warning(f"{backend_name} evaluation failed: {e}")
            error = str(e)
            print(f"  {backend_name} failed: {e}")
            continue

    if results is None:
        # All backends failed
        results = {
            "score": 0.0,
            "metric": "accuracy",
            "details": json.dumps({"error": error or "all_backends_failed"}),
            "backend": "none",
            "status": "failed",
            "model": model,
            "tasks": tasks,
        }
    else:
        results["status"] = "completed"
        results["model"] = model
        results["tasks"] = tasks

    # Store in database
    eval_id = db.store_evaluation_result({
        "training_run_id": None,
        "model": model,
        "task": tasks if isinstance(tasks, str) else ",".join(tasks),
        "score": results.get("score", 0.0),
        "metric": results.get("metric", "accuracy"),
        "details": results.get("details", ""),
    })

    results["evaluation_id"] = eval_id

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Backend:  {results.get('backend', 'none')}")
    print(f"  Score:    {results.get('score', 0.0):.4f}")
    print(f"  Metric:   {results.get('metric', 'accuracy')}")
    print(f"  Status:   {results.get('status', 'unknown')}")
    print(f"{'='*60}\n")

    db.close()
    return results


def compare_models(
    base_model: str,
    tuned_model: str,
    tasks: str = "mmlu_stem",
    num_fewshot: int = 5,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare a base model with a fine-tuned model.

    Args:
        base_model: Base model name or path
        tuned_model: Fine-tuned model name or path
        tasks: Evaluation task(s)
        num_fewshot: Number of few-shot examples
        adapter_path: Path to adapter weights for the tuned model

    Returns:
        dict with keys:
            - base_score: Score of the base model
            - tuned_score: Score of the tuned model
            - improvement: Difference (tuned - base)
            - improvement_pct: Percentage improvement
            - tasks: Task name(s)
            - base_results: Full results dict for base model
            - tuned_results: Full results dict for tuned model
    """
    print(f"\n{'='*60}")
    print(f"  MindForge Model Comparison")
    print(f"{'='*60}")
    print(f"  Base model:   {base_model}")
    print(f"  Tuned model:  {tuned_model}")
    print(f"  Tasks:        {tasks}")
    print(f"{'='*60}\n")

    # Evaluate base model
    print("  --- Evaluating Base Model ---")
    base_results = evaluate_model(
        model=base_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        adapter_path=None,
    )

    print("\n  --- Evaluating Tuned Model ---")
    tuned_results = evaluate_model(
        model=tuned_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        adapter_path=adapter_path,
    )

    base_score = base_results.get("score", 0.0)
    tuned_score = tuned_results.get("score", 0.0)
    improvement = tuned_score - base_score
    improvement_pct = (improvement / base_score * 100) if base_score > 0 else 0.0

    comparison = {
        "base_score": base_score,
        "tuned_score": tuned_score,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "tasks": tasks,
        "base_results": base_results,
        "tuned_results": tuned_results,
    }

    print(f"\n{'='*60}")
    print(f"  Comparison Results")
    print(f"{'='*60}")
    print(f"  Base model score:   {base_score:.4f}")
    print(f"  Tuned model score:  {tuned_score:.4f}")
    print(f"  Improvement:        {improvement:+.4f} ({improvement_pct:+.1f}%)")
    print(f"{'='*60}\n")

    return comparison
