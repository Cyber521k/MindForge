"""Standalone quantization module for re-quantizing existing MLX models.

Uses mlx_lm.convert() with quantize parameters to re-quantize
models that have already been converted to MLX format.
"""

import os
import uuid
import logging
import time

from mindforge.convert.converter import _get_model_size_gb, _get_db

logger = logging.getLogger(__name__)

# Import mlx_lm lazily but store at module level for mockability
try:
    from mlx_lm import convert as mlx_convert
    from mlx_lm import upload as mlx_upload
except ImportError:
    mlx_convert = None
    mlx_upload = None

# Default directory for quantized models
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "mindforge-data", "models"
)

# Valid bit depths
VALID_BITS = [2, 3, 4, 6, 8]


def quantize_model(
    source_path,
    bits=4,
    group_size=64,
    output_dir=None,
    upload_repo=None,
):
    """Re-quantize an existing MLX model.

    Args:
        source_path: Path to the existing MLX model directory
        bits: Target quantization bits - 2, 3, 4, 6, or 8 (default 4)
        group_size: Quantization group size (default 64)
        output_dir: Output directory (default ~/mindforge-data/models/)
        upload_repo: HuggingFace repo to upload to (default None)

    Returns:
        dict with keys:
            - id: unique quantization ID
            - source_path: path to the source model
            - output_path: path to the quantized model
            - bit_depth: target bit depth
            - group_size: quantization group size
            - model_size_gb: size of quantized model in GB
            - uploaded_to_hf: whether uploaded to HF
            - hf_repo: HF repo name if uploaded
            - quantized_at: timestamp
    """
    # Set default output directory
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Validate bits
    if bits not in VALID_BITS:
        raise ValueError(
            f"Invalid bits={bits}. Must be one of {VALID_BITS}"
        )

    # Validate source path exists
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Source model not found: {source_path}")

    # Derive output name
    source_name = os.path.basename(os.path.normpath(source_path))
    # Strip existing quantization suffix if present
    for suffix in ["-2bit", "-3bit", "-4bit", "-6bit", "-8bit", "-MLX", "-full"]:
        if source_name.endswith(suffix):
            source_name = source_name[: -len(suffix)]
            break

    output_name = f"{source_name}-MLX-{bits}bit"
    output_path = os.path.join(output_dir, output_name)

    logger.info(f"Quantizing {source_path} to {bits}-bit...")
    logger.info(f"  Group size: {group_size}")
    logger.info(f"  Output: {output_path}")

    # Call mlx_lm.convert() to re-quantize
    from mlx_lm import convert as mlx_convert

    mlx_convert(
        source_path,
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
        mlx_path=output_path,
    )

    # Upload to HuggingFace if requested
    uploaded_to_hf = False
    hf_repo = None
    if upload_repo:
        logger.info(f"Uploading to HuggingFace: {upload_repo}")
        try:
            from mlx_lm import upload
            upload(upload_repo, output_path)
            uploaded_to_hf = True
            hf_repo = upload_repo
        except Exception as e:
            logger.warning(f"Upload failed: {e}")

    # Calculate model size
    model_size_gb = _get_model_size_gb(output_path)

    # Build result
    quant_id = str(uuid.uuid4())
    quantized_at = time.time()

    result = {
        "id": quant_id,
        "source_path": source_path,
        "output_path": output_path,
        "bit_depth": bits,
        "group_size": group_size,
        "model_size_gb": model_size_gb,
        "uploaded_to_hf": uploaded_to_hf,
        "hf_repo": hf_repo,
        "quantized_at": quantized_at,
    }

    # Store in database
    try:
        db = _get_db()
        db.store_quantized_model(result)
        db.close()
    except Exception as e:
        logger.warning(f"Failed to store quantization record in DB: {e}")

    logger.info(f"✓ Quantization complete: {output_path} ({model_size_gb} GB)")
    return result


def list_quantized_models():
    """List all quantized models from the database.

    Returns:
        list of dicts with quantization info
    """
    try:
        db = _get_db()
        models = db.get_quantized_models()
        db.close()
        return models
    except Exception as e:
        logger.warning(f"Failed to list quantized models: {e}")
        return []
