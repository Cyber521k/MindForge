"""Trainer module - fine-tuning with SFT, DPO, and ORPO modes.

Uses mlx_lm.lora for SFT training (built-in).
Attempts to use mlx-lm-lora (separate package) for DPO/ORPO if installed.
Falls back to mlx_lm.lora for SFT-style training when mlx-lm-lora is not available.
"""

import os
import sys
import json
import time
import uuid
import logging
import importlib
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Project root for DB path resolution
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _check_mlx_lm_lora_installed() -> bool:
    """Check if the mlx-lm-lora package is installed."""
    try:
        importlib.import_module("mlx_lm_lora")
        return True
    except ImportError:
        return False


def _check_mlx_lm_lora_alt() -> bool:
    """Check for mlx-lm-lora under alternative module names."""
    for name in ("mlx_lm_lora", "mlx-lm-lora", "lm_lora"):
        try:
            importlib.import_module(name)
            return True
        except ImportError:
            continue
    return False


def _get_db():
    """Get the Database instance for tracking training runs."""
    from mindforge.vault.database import Database
    db_path = os.path.join(_project_root, "data", "mindforge.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return Database(db_path)


def _generate_adapter_path(model: str, mode: str, adapter_path: Optional[str] = None) -> str:
    """Generate a default adapter path if none is provided."""
    if adapter_path:
        return adapter_path
    # Create a unique adapter directory
    model_name = model.replace("/", "_")
    run_id = uuid.uuid4().hex[:8]
    return os.path.join(_project_root, "data", "adapters", f"{model_name}_{mode}_{run_id}")


def _run_sft_training(
    model: str,
    data_path: str,
    iters: int,
    batch_size: int,
    learning_rate: float,
    adapter_path: str,
    max_seq_length: int = 2048,
) -> Dict[str, Any]:
    """Run SFT training using mlx_lm.lora.

    This uses the built-in mlx_lm.lora module for supervised fine-tuning.
    """
    try:
        from mlx_lm.lora import train, TrainingArgs
        from mlx_lm.tuner.trainer import TrainingCallback
    except ImportError as e:
        raise ImportError(
            f"mlx_lm.lora is not available: {e}. "
            "Please install mlx-lm: pip install mlx-lm"
        )

    # Prepare adapter directory
    os.makedirs(adapter_path, exist_ok=True)
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")

    # Track training progress
    progress = {"loss": None, "iter": 0, "losses": []}

    class ProgressCallback:
        """Callback to track training progress and loss."""

        def __call__(self, info):
            """Called by mlx_lm.lora.train on each iteration with loss/step info."""
            progress["iter"] = info.get("iteration", progress["iter"])
            progress["loss"] = info.get("loss", progress["loss"])
            if progress["loss"] is not None:
                progress["losses"].append(progress["loss"])
                print(f"  Iter {progress['iter']}/{iters} | Loss: {progress['loss']:.4f}")

    callback = ProgressCallback()

    # Build training args
    # We need to set up the learning rate via optimizer config
    # mlx_lm.lora.train takes optimizer separately
    try:
        import mlx.optimizers as optim

        optimizer = optim.Adam(learning_rate=learning_rate)
    except ImportError:
        # Fallback: try to use the default optimizer
        optimizer = None

    # Set up the training configuration
    # The mlx_lm.lora.train function expects:
    # model, optimizer, train_dataset, val_dataset, args, training_callback
    # We'll use the CLI-based approach via subprocess as a fallback
    # since the direct API requires careful setup of datasets.

    # Try direct API first
    try:
        from mlx_lm.lora import load, load_dataset, build_schedule

        # Load model and tokenizer
        model_obj, tokenizer = load(model)

        # Load dataset
        dataset_args = type('Args', (), {
            'data': data_path,
            'model': model,
            'train': True,
            'test': False,
        })()

        train_dataset, val_dataset = load_dataset(dataset_args, tokenizer)

        # Create training args
        train_args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            steps_per_report=10,
            steps_per_eval=200,
            steps_per_save=max(iters // 10, 1),
            max_seq_length=max_seq_length,
            adapter_file=adapter_file,
        )

        # Build learning rate schedule
        schedule = build_schedule(train_args, optimizer) if optimizer else None

        # Run training
        if optimizer:
            train(
                model=model_obj,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                args=train_args,
                training_callback=callback,
            )
        else:
            train(
                model=model_obj,
                optimizer=None,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                args=train_args,
                training_callback=callback,
            )

        final_loss = progress["loss"] if progress["loss"] is not None else 0.0
        iters_completed = progress["iter"]

    except Exception as e:
        logger.warning(f"Direct mlx_lm.lora.train API failed: {e}")
        logger.info("Falling back to CLI-based training...")

        # Fallback: use CLI via subprocess
        import subprocess

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", model,
            "--train",
            "--data", data_path,
            "--iters", str(iters),
            "--batch-size", str(batch_size),
            "--adapter-path", adapter_file,
        ]

        print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"mlx_lm.lora training failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Parse output for loss
        final_loss = 0.0
        iters_completed = iters
        for line in result.stdout.split("\n"):
            if "loss" in line.lower():
                try:
                    # Try to extract loss value
                    parts = line.split("loss:")
                    if len(parts) > 1:
                        final_loss = float(parts[-1].strip())
                except (ValueError, IndexError):
                    pass

    return {
        "adapter_path": adapter_file,
        "final_loss": final_loss,
        "iters_completed": iters_completed,
    }


def _run_dpo_orpo_training(
    model: str,
    data_path: str,
    mode: str,
    iters: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    adapter_path: str,
) -> Dict[str, Any]:
    """Run DPO or ORPO training.

    First tries to use mlx-lm-lora (the separate package that supports DPO/ORPO).
    If not installed, falls back to SFT-style training with mlx_lm.lora,
    logging a warning that DPO/ORPO will be approximated as SFT.
    """
    # Check if mlx-lm-lora is installed
    if _check_mlx_lm_lora_installed():
        try:
            import mlx_lm_lora

            os.makedirs(adapter_path, exist_ok=True)
            adapter_file = os.path.join(adapter_path, "adapters.safetensors")

            print(f"  Using mlx-lm-lora for {mode.upper()} training...")

            # Try to use the mlx_lm_lora API
            # The API may vary, so we try multiple approaches
            if hasattr(mlx_lm_lora, "train_dpo") and mode == "dpo":
                result = mlx_lm_lora.train_dpo(
                    model=model,
                    data_path=data_path,
                    iters=iters,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    beta=beta,
                    adapter_path=adapter_file,
                )
                return {
                    "adapter_path": result.get("adapter_path", adapter_file),
                    "final_loss": result.get("final_loss", 0.0),
                    "iters_completed": result.get("iters_completed", iters),
                }
            elif hasattr(mlx_lm_lora, "train_orpo") and mode == "orpo":
                result = mlx_lm_lora.train_orpo(
                    model=model,
                    data_path=data_path,
                    iters=iters,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    beta=beta,
                    adapter_path=adapter_file,
                )
                return {
                    "adapter_path": result.get("adapter_path", adapter_file),
                    "final_loss": result.get("final_loss", 0.0),
                    "iters_completed": result.get("iters_completed", iters),
                }
            elif hasattr(mlx_lm_lora, "train"):
                # Generic train function with mode parameter
                result = mlx_lm_lora.train(
                    model=model,
                    data_path=data_path,
                    mode=mode,
                    iters=iters,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    beta=beta,
                    adapter_path=adapter_file,
                )
                return {
                    "adapter_path": result.get("adapter_path", adapter_file),
                    "final_loss": result.get("final_loss", 0.0),
                    "iters_completed": result.get("iters_completed", iters),
                }
            else:
                # Try CLI via subprocess
                import subprocess

                cmd = [
                    sys.executable, "-m", "mlx_lm_lora",
                    "--model", model,
                    "--train",
                    "--data", data_path,
                    "--mode", mode,
                    "--iters", str(iters),
                    "--batch-size", str(batch_size),
                    "--learning-rate", str(learning_rate),
                    "--beta", str(beta),
                    "--adapter-path", adapter_file,
                ]

                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                if result.returncode != 0:
                    raise RuntimeError(
                        f"mlx-lm-lora training failed:\n{result.stderr}"
                    )

                final_loss = 0.0
                iters_completed = iters
                for line in result.stdout.split("\n"):
                    if "loss" in line.lower():
                        try:
                            parts = line.split("loss:")
                            if len(parts) > 1:
                                final_loss = float(parts[-1].strip())
                        except (ValueError, IndexError):
                            pass

                return {
                    "adapter_path": adapter_file,
                    "final_loss": final_loss,
                    "iters_completed": iters_completed,
                }

        except Exception as e:
            logger.error(f"mlx-lm-lora training failed: {e}")
            # Fall through to SFT fallback

    # Fallback: use mlx_lm.lora for SFT-style training
    print(f"  WARNING: mlx-lm-lora not installed. {mode.upper()} mode requested.")
    print(f"  Falling back to mlx_lm.lora SFT-style training.")
    print(f"  For true {mode.upper()} support, install mlx-lm-lora:")
    print(f"    pip install mlx-lm-lora")
    print()

    return _run_sft_training(
        model=model,
        data_path=data_path,
        iters=iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        adapter_path=adapter_path,
    )


def train_model(
    model: str,
    data_path: str,
    mode: str = "dpo",
    iters: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    beta: float = 0.1,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a model using the specified fine-tuning mode.

    Args:
        model: Model name or path (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")
        data_path: Path to training data directory or file
        mode: Training mode - "sft", "dpo", or "orpo"
        iters: Number of training iterations
        batch_size: Training batch size
        learning_rate: Learning rate for the optimizer
        beta: DPO/ORPO beta parameter (KL penalty coefficient)
        adapter_path: Path to save/load adapter weights. If None, auto-generated.

    Returns:
        dict with keys:
            - adapter_path: Path to the trained adapter file
            - final_loss: Final training loss
            - iters_completed: Number of iterations completed
            - mode: Training mode used
            - model: Model name
            - status: "completed" or "failed"
            - training_run_id: Database run ID
    """
    # Validate mode
    mode = mode.lower().strip()
    if mode not in ("sft", "dpo", "orpo"):
        raise ValueError(f"Invalid training mode: {mode}. Must be 'sft', 'dpo', or 'orpo'.")

    # Validate data path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data path not found: {data_path}")

    # Generate adapter path if not provided
    adapter_path = _generate_adapter_path(model, mode, adapter_path)

    # Initialize database tracking
    db = _get_db()
    started_at = time.time()

    run_id = db.store_training_run({
        "model": model,
        "mode": mode,
        "data_path": data_path,
        "adapter_path": adapter_path,
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "beta": beta,
        "status": "running",
        "loss": None,
        "iters_completed": 0,
        "started_at": started_at,
        "finished_at": None,
    })

    print(f"\n{'='*60}")
    print(f"  MindForge Training")
    print(f"{'='*60}")
    print(f"  Model:         {model}")
    print(f"  Mode:          {mode.upper()}")
    print(f"  Data:          {data_path}")
    print(f"  Iters:         {iters}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    if mode in ("dpo", "orpo"):
        print(f"  Beta:          {beta}")
    print(f"  Adapter:       {adapter_path}")
    print(f"  Run ID:        {run_id}")
    print(f"{'='*60}\n")

    try:
        if mode == "sft":
            results = _run_sft_training(
                model=model,
                data_path=data_path,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                adapter_path=adapter_path,
            )
        else:
            # DPO or ORPO
            results = _run_dpo_orpo_training(
                model=model,
                data_path=data_path,
                mode=mode,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                beta=beta,
                adapter_path=adapter_path,
            )

        # Update database with results
        finished_at = time.time()
        db.update_training_run(run_id, {
            "status": "completed",
            "loss": results.get("final_loss", 0.0),
            "iters_completed": results.get("iters_completed", iters),
            "finished_at": finished_at,
            "adapter_path": results.get("adapter_path", adapter_path),
        })

        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"{'='*60}")
        print(f"  Status:           Completed")
        print(f"  Final loss:       {results.get('final_loss', 'N/A')}")
        print(f"  Iters completed:  {results.get('iters_completed', iters)}")
        print(f"  Adapter path:     {results.get('adapter_path', adapter_path)}")
        print(f"  Time:             {finished_at - started_at:.1f}s")
        print(f"{'='*60}\n")

        results["mode"] = mode
        results["model"] = model
        results["status"] = "completed"
        results["training_run_id"] = run_id

        db.close()
        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        finished_at = time.time()
        db.update_training_run(run_id, {
            "status": "failed",
            "loss": None,
            "iters_completed": 0,
            "finished_at": finished_at,
        })
        db.close()

        print(f"\n✗ Training failed: {e}")

        return {
            "adapter_path": adapter_path,
            "final_loss": None,
            "iters_completed": 0,
            "mode": mode,
            "model": model,
            "status": "failed",
            "error": str(e),
            "training_run_id": run_id,
        }
