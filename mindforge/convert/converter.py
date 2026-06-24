"""Model conversion module using mlx_lm.convert() API.

Converts HuggingFace models to MLX format with optional quantization.
Converted models are stored in ~/mindforge-data/models/ and tracked in SQLite.
"""

import os
import uuid
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Import mlx_lm lazily but store at module level for mockability
try:
    from mlx_lm import convert as mlx_convert
    from mlx_lm import upload as mlx_upload
except ImportError:
    mlx_convert = None
    mlx_upload = None

# Default directory for converted models
DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), "mindforge-data", "models"
)

# Valid quantization bit depths
VALID_Q_BITS = [2, 3, 4, 6, 8]


def _parse_quantize_flag(quantize):
    """Parse the --quantize CLI flag into (quantize_bool, q_bits).

    Args:
        quantize: True, False, None, or a string like '4bit', '8bit', 'full', 'none'

    Returns:
        tuple: (quantize_bool, q_bits) where q_bits is int or None
    """
    if quantize is True:
        return True, 4  # default to 4-bit
    if quantize is False or quantize is None:
        return False, None
    if isinstance(quantize, str):
        q_lower = quantize.lower().strip()
        if q_lower in ("none", "full", "fp16", "bf16"):
            return False, None
        if q_lower.endswith("bit"):
            try:
                bits = int(q_lower.replace("bit", ""))
                if bits in VALID_Q_BITS:
                    return True, bits
            except ValueError:
                pass
        # Try direct int parse
        try:
            bits = int(q_lower)
            if bits in VALID_Q_BITS:
                return True, bits
        except ValueError:
            pass
    # Default fallback
    return True, 4


def _get_model_size_gb(path):
    """Calculate total size of a model directory in GB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 ** 3), 2)


def _get_db():
    """Get a Database instance."""
    from mindforge.vault.database import Database
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(project_root, "data", "mindforge.db")
    return Database(db_path)


def convert_model(
    source_repo,
    quantize=True,
    q_bits=4,
    q_group_size=64,
    output_dir=None,
    upload_repo=None,
):
    """Convert a HuggingFace model to MLX format with optional quantization.

    Args:
        source_repo: HuggingFace repo ID (e.g., 'mistralai/Mistral-7B-Instruct-v0.3')
        quantize: Whether to quantize the model (default True)
        q_bits: Quantization bits - 2, 3, 4, 6, or 8 (default 4)
        q_group_size: Quantization group size (default 64)
        output_dir: Output directory (default ~/mindforge-data/models/)
        upload_repo: HuggingFace repo to upload to (default None)

    Returns:
        dict with keys:
            - id: unique conversion ID
            - source_repo: the source HF repo
            - local_path: path to the converted model
            - quantization: quantization description (e.g., '4bit' or 'full')
            - model_size_gb: size of converted model in GB
            - uploaded_to_hf: whether uploaded to HF
            - hf_repo: HF repo name if uploaded
            - converted_at: timestamp
    """
    # Set default output directory
    if output_dir is None:
        output_dir = DEFAULT_MODELS_DIR

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Validate q_bits
    if quantize and q_bits not in VALID_Q_BITS:
        raise ValueError(
            f"Invalid q_bits={q_bits}. Must be one of {VALID_Q_BITS}"
        )

    # Derive model name from repo
    model_name = source_repo.split("/")[-1]
    quant_suffix = f"-{q_bits}bit" if quantize else "-full"
    model_dir_name = f"{model_name}-MLX{quant_suffix}"
    local_path = os.path.join(output_dir, model_dir_name)

    logger.info(f"Converting {source_repo} to MLX format...")
    if quantize:
        logger.info(f"  Quantization: {q_bits}-bit (group size {q_group_size})")
    else:
        logger.info("  No quantization (full precision)")

    # Call mlx_lm.convert() API
    mlx_convert(
        source_repo,
        quantize=quantize,
        q_bits=q_bits if quantize else None,
        q_group_size=q_group_size if quantize else None,
        mlx_path=local_path,
    )

    # Upload to HuggingFace if requested
    uploaded_to_hf = False
    hf_repo = None
    if upload_repo:
        logger.info(f"Uploading to HuggingFace: {upload_repo}")
        try:
            mlx_upload(upload_repo, local_path)
            uploaded_to_hf = True
            hf_repo = upload_repo
        except Exception as e:
            logger.warning(f"Upload failed: {e}")

    # Calculate model size
    model_size_gb = _get_model_size_gb(local_path)

    # Build result
    conversion_id = str(uuid.uuid4())
    quant_label = f"{q_bits}bit" if quantize else "full"
    converted_at = time.time()

    result = {
        "id": conversion_id,
        "source_repo": source_repo,
        "local_path": local_path,
        "quantization": quant_label,
        "model_size_gb": model_size_gb,
        "uploaded_to_hf": uploaded_to_hf,
        "hf_repo": hf_repo,
        "converted_at": converted_at,
    }

    # Store in database
    try:
        db = _get_db()
        db.store_converted_model(result)
        db.close()
    except Exception as e:
        logger.warning(f"Failed to store conversion record in DB: {e}")

    logger.info(f"✓ Conversion complete: {local_path} ({model_size_gb} GB)")
    return result


def list_converted_models():
    """List all converted models from the database.

    Returns:
        list of dicts with conversion info
    """
    try:
        db = _get_db()
        models = db.get_converted_models()
        db.close()
        return models
    except Exception as e:
        logger.warning(f"Failed to list converted models: {e}")
        return []
