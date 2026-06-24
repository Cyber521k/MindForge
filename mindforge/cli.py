#!/usr/bin/env python3
"""MindForge CLI - Main entry point."""

import argparse
import sys
import os
import json
import logging

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mindforge.hardware.detector import detect_hardware, format_hardware_info
from mindforge.hardware.api_keys import detect_available_apis, format_api_info


def cmd_detect(args):
    """Run hardware detection."""
    hw = detect_hardware()
    print(format_hardware_info(hw))

    apis = detect_available_apis()
    print(format_api_info(apis))

    # Exo cluster detection (Phase 6)
    try:
        from mindforge.hardware.exo_detector import detect_exo, get_cluster_info, format_cluster_info
        exo = detect_exo()

        if exo.get("running"):
            # Get detailed cluster info if exo is running
            cluster_info = get_cluster_info(exo.get("api_url"))
            print("")
            print(format_cluster_info(cluster_info))

            # Also store in database
            try:
                from mindforge.vault.database import Database
                import os as _os
                db_path = _os.path.join(_project_root, "data", "mindforge.db")
                _os.makedirs(_os.path.dirname(db_path), exist_ok=True)
                db = Database(db_path)
                exo_status = {**exo, **cluster_info}
                db.store_exo_status(exo_status)
                db.close()
            except Exception:
                pass

            if exo.get("peer_count", 0) > 0:
                print(f"\n  ✓ Exo cluster active with {exo['peer_count']} peer(s)")
                print(f"    Cluster-powered models available via 'mindforge models'")
            else:
                print(f"\n  ⚠ Exo running but no peers connected")
                print(f"    Falling back to single MLX")
        elif exo.get("installed"):
            print(f"\n=== Exo Cluster ===")
            print(f"  Status: Installed but not running")
            print(f"  Start exo to enable cluster inference")
        # If not installed, stay silent (fallback to single MLX)
    except ImportError:
        pass

    # Show recommended settings
    print("\n=== Recommendations ===")
    chip = hw.get("chip", "")
    mem_gb = hw.get("memory_gb", 0)

    if "M1" in chip or "M2" in chip or "M3" in chip or "M4" in chip:
        if mem_gb >= 16:
            print(f"  ✓ MLX models up to ~8B parameters (4-bit) should work well with {mem_gb} GB")
        else:
            print(f"  ⚠ With {mem_gb} GB, stick to smaller models (3B or less)")
    else:
        print("  ⚠ Apple Silicon not detected. MLX may not be available.")

    return 0


def cmd_models(args):
    """Show available models based on hardware and API keys."""
    from mindforge.hardware.model_list import get_available_models, format_model_list

    model_info = get_available_models()

    # Exo cluster integration (Phase 6)
    try:
        from mindforge.hardware.exo_detector import detect_exo, get_cluster_info, format_cluster_info
        exo = detect_exo()
        if exo.get("running") and exo.get("peer_count", 0) > 0:
            cluster_info = get_cluster_info(exo.get("api_url"))
            model_info["exo_cluster"] = cluster_info
            # Re-classify models: larger models become available with cluster memory
            cluster_mem = cluster_info.get("total_usable_gb", 0)
            from mindforge.hardware.model_list import get_memory_tier
            cluster_tier = get_memory_tier(cluster_mem)
            model_info["cluster_memory_tier"] = cluster_tier
            model_info["cluster_memory_gb"] = cluster_mem
    except Exception:
        pass

    print(format_model_list(model_info))

    # Show cluster-powered models if exo active
    if model_info.get("exo_cluster"):
        cluster_info = model_info["exo_cluster"]
        print("")
        try:
            from mindforge.hardware.exo_detector import format_cluster_info
            print(format_cluster_info(cluster_info))
        except ImportError:
            pass
        print(f"\n  ✓ Cluster-powered models unlocked!")
        print(f"    Effective memory tier: {model_info.get('cluster_memory_tier', '?')}")
        print(f"    (was tier {model_info.get('memory_tier', '?')} on single device)")

    return 0


def cmd_probe(args):
    """Run model probing."""
    from mindforge.probe.engine import ProbeEngine

    # Handle --tier all
    if args.tier == "all":
        tier = "all"
    else:
        tier = int(args.tier)

    engine = ProbeEngine(
        model_name=args.model,
        subject=args.subject,
        tier=tier,
        limit=args.limit,
        judge_model=args.judge_model,
    )

    results = engine.run()

    if "error" in results:
        print(f"✗ Probe failed: {results.get('error', 'Unknown error')}")
        print(f"  Tip: Ensure the model name is correct and mlx-lm is installed (pip install mlx-lm).")
        print(f"  For cloud models, ensure the relevant API key is set (e.g., OPENAI_API_KEY).")
        return 1

    print(f"\n✓ Probe complete. Output: {results.get('output_path', 'N/A')}")
    print(f"  DPO entries generated: {results.get('dpo_entries', 0)}")
    return 0


def cmd_review(args):
    """Run review session."""
    from mindforge.vault.database import Database
    from mindforge.vault.review import review_session

    db_path = os.path.join(_project_root, "data", "mindforge.db")
    db = Database(db_path)

    review_session(db, limit=args.limit)

    db.close()
    return 0


def cmd_format(args):
    """Format training data."""
    # Load input data
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print(f"  Tip: Check the file path is correct. Use 'mindforge format --input <file>' to specify the path.")
        print(f"  Supported formats: JSON (.json) or JSONL (.jsonl)")
        return 1

    with open(args.input, "r") as f:
        if args.input.endswith(".jsonl"):
            entries = [json.loads(line) for line in f if line.strip()]
        else:
            entries = json.load(f)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Select formatter
    fmt = args.format.lower()
    if fmt == "dpo":
        from mindforge.format.dpo import format_dpo_batch, write_dpo_jsonl
        formatted = format_dpo_batch(entries)
        write_dpo_jsonl(formatted, args.output)
    elif fmt == "alpaca":
        from mindforge.format.alpaca import format_alpaca_batch
        formatted = format_alpaca_batch(entries)
        with open(args.output, "w") as f:
            json.dump(formatted, f, indent=2)
    elif fmt == "chatml":
        from mindforge.format.chatml import format_chatml_batch
        formatted = format_chatml_batch(entries)
        with open(args.output, "w") as f:
            for entry in formatted:
                f.write(json.dumps(entry) + "\n")
    elif fmt == "completion":
        from mindforge.format.completion import format_completion_batch
        formatted = format_completion_batch(entries)
        with open(args.output, "w") as f:
            for entry in formatted:
                f.write(json.dumps(entry) + "\n")
    elif fmt == "openai_messages":
        from mindforge.format.openai_messages import format_openai_messages_batch
        formatted = format_openai_messages_batch(entries)
        with open(args.output, "w") as f:
            for entry in formatted:
                f.write(json.dumps(entry) + "\n")
    elif fmt == "template_free":
        from mindforge.format.template_free import format_template_free_batch
        formatted = format_template_free_batch(entries)
        with open(args.output, "w") as f:
            for entry in formatted:
                f.write(json.dumps(entry) + "\n")
    else:
        print(f"ERROR: Unknown format: {fmt}")
        print(f"  Supported formats: dpo, alpaca, chatml, completion, openai_messages, template_free")
        return 1

    print(f"✓ Formatted {len(formatted)} entries as {fmt} -> {args.output}")
    return 0


def cmd_convert(args):
    """Convert a HuggingFace model to MLX format."""
    from mindforge.convert.converter import convert_model, _parse_quantize_flag

    # Parse --quantize flag
    quantize_val = args.quantize
    if quantize_val and quantize_val.lower() not in ("none", "full"):
        quantize_bool, q_bits = _parse_quantize_flag(quantize_val)
    else:
        quantize_bool = False
        q_bits = None

    print(f"=== Model Conversion ===")
    print(f"  Source:     {args.source}")
    print(f"  Quantize:   {quantize_val or 'none'}")
    if quantize_bool:
        print(f"  Bits:       {q_bits}-bit")
        print(f"  Group size: {args.group_size}")
    if args.upload_repo:
        print(f"  Upload to:  {args.upload_repo}")
    print()

    try:
        result = convert_model(
            source_repo=args.source,
            quantize=quantize_bool,
            q_bits=q_bits if quantize_bool else 4,
            q_group_size=args.group_size,
            upload_repo=args.upload_repo,
        )
        print(f"\n✓ Conversion complete!")
        print(f"  Local path:  {result['local_path']}")
        print(f"  Size:        {result['model_size_gb']} GB")
        print(f"  Quantization: {result['quantization']}")
        if result.get("uploaded_to_hf"):
            print(f"  Uploaded to: {result['hf_repo']}")
        return 0
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        print(f"  Tip: Ensure mlx-lm is installed (pip install mlx-lm) and the HuggingFace repo name is correct.")
        print(f"  Example: mindforge convert --source mistralai/Mistral-7B-Instruct-v0.3 --quantize 4bit")
        return 1


def cmd_quantize(args):
    """Re-quantize an existing MLX model."""
    from mindforge.convert.quantizer import quantize_model

    print(f"=== Model Quantization ===")
    print(f"  Source:     {args.model}")
    print(f"  Bits:       {args.bits}-bit")
    print(f"  Group size: {args.group_size}")
    if args.upload_repo:
        print(f"  Upload to:  {args.upload_repo}")
    print()

    try:
        result = quantize_model(
            source_path=args.model,
            bits=args.bits,
            group_size=args.group_size,
            upload_repo=args.upload_repo,
        )
        print(f"\n✓ Quantization complete!")
        print(f"  Output path: {result['output_path']}")
        print(f"  Size:        {result['model_size_gb']} GB")
        print(f"  Bit depth:   {result['bit_depth']}-bit")
        if result.get("uploaded_to_hf"):
            print(f"  Uploaded to: {result['hf_repo']}")
        return 0
    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        print(f"  Tip: Ensure the model path points to a valid MLX model directory (must contain config.json).")
        print(f"  Ensure mlx-lm is installed: pip install mlx-lm")
        return 1


def cmd_train(args):
    """Run model fine-tuning."""
    from mindforge.train.trainer import train_model

    try:
        results = train_model(
            model=args.model,
            data_path=args.data,
            mode=args.mode,
            iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            beta=args.beta,
            adapter_path=args.adapter_path,
        )

        if results.get("status") == "failed":
            return 1
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print(f"  Tip: Ensure mlx-lm is installed (pip install mlx-lm) and the model name is correct.")
        print(f"  For DPO/ORPO, install mlx-lm-lora: pip install mlx-lm-lora")
        print(f"  Training data must be in JSONL format with prompt/chosen/rejected fields.")
        return 1


def cmd_evaluate(args):
    """Run model evaluation."""
    from mindforge.evaluate.evaluator import evaluate_model, compare_models

    try:
        if args.compare:
            # Compare two models
            results = compare_models(
                base_model=args.model,
                tuned_model=args.compare,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                adapter_path=args.adapter_path,
            )
        else:
            results = evaluate_model(
                model=args.model,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                adapter_path=args.adapter_path,
            )

            if results.get("status") == "failed":
                return 1
        return 0
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        print(f"  Tip: For lm-eval-harness, install it: pip install lm-eval")
        print(f"  For mlx evaluation, ensure mlx-lm is installed: pip install mlx-lm")
        print(f"  The model path must point to a valid MLX model or HuggingFace repo.")
        return 1


def cmd_ingest_pdf(args):
    """Ingest a PDF file and generate training data."""
    from mindforge.ingest.pdf_extractor import extract_pdf, chunk_text, generate_qa_pairs
    from mindforge.ingest.qa_generator import format_qa_as_dpo
    from mindforge.vault.database import Database
    from mindforge.format.dpo import write_dpo_jsonl

    if not os.path.exists(args.file):
        print(f"ERROR: PDF file not found: {args.file}")
        print(f"  Tip: Provide an absolute or relative path to a .pdf file.")
        print(f"  Example: mindforge ingest-pdf --file /path/to/document.pdf")
        return 1

    print(f"=== PDF Ingestion ===")
    print(f"  File:     {args.file}")
    print(f"  Subject:  {args.subject or 'auto'}")
    print(f"  Format:   {args.format}")
    print()

    # Step 1: Extract text
    print("Extracting text from PDF...")
    try:
        result = extract_pdf(args.file)
    except Exception as e:
        print(f"✗ PDF extraction failed: {e}")
        print(f"  Tip: Ensure the file is a valid PDF. Install pymupdf if needed: pip install pymupdf")
        return 1

    print(f"  Pages:      {result['metadata']['page_count']}")
    print(f"  Words:       {result['metadata']['word_count']}")
    print(f"  Content hash: {result['metadata']['content_hash'][:16]}...")
    print()

    # Step 2: Chunk text
    print("Chunking text...")
    chunks = chunk_text(result["text"])
    print(f"  Chunks: {len(chunks)}")
    print()

    # Step 3: Generate Q&A pairs
    print("Generating Q&A pairs (heuristic mode, no adapter)...")
    qa_pairs = generate_qa_pairs(chunks, subject=args.subject, adapter=None)
    print(f"  Q&A pairs: {len(qa_pairs)}")
    print()

    if not qa_pairs:
        print("⚠ No Q&A pairs generated. The PDF may have insufficient text structure.")
        return 0

    # Step 4: Format output
    fmt = args.format.lower()
    output_dir = os.path.join(_project_root, "data", "training-data", fmt)
    os.makedirs(output_dir, exist_ok=True)

    if fmt == "dpo":
        dpo_entries = format_qa_as_dpo(qa_pairs)
        output_path = os.path.join(output_dir, "ingest_pdf.jsonl")
        with open(output_path, "w") as f:
            for entry in dpo_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"✓ DPO training data written to: {output_path}")
        print(f"  Total entries: {len(dpo_entries)}")
    else:
        # Write raw Q&A pairs
        output_path = os.path.join(output_dir, "ingest_pdf.json")
        with open(output_path, "w") as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"✓ Q&A pairs written to: {output_path}")
        print(f"  Total pairs: {len(qa_pairs)}")

    # Step 5: Store in database
    try:
        db_path = os.path.join(_project_root, "data", "mindforge.db")
        db = Database(db_path)
        db.store_pdf_source({
            "filename": result["metadata"]["filename"],
            "file_path": result["metadata"]["file_path"],
            "page_count": result["metadata"]["page_count"],
            "word_count": result["metadata"]["word_count"],
            "content_hash": result["metadata"]["content_hash"],
        })
        db.close()
        print(f"  Source recorded in database.")
    except Exception as e:
        print(f"  ⚠ Failed to record in database: {e}")

    return 0


def cmd_ingest_web(args):
    """Ingest web content from a URL and generate training data."""
    from mindforge.ingest.web_extractor import extract_url, crawl_site
    from mindforge.ingest.sanitizer import sanitize_content
    from mindforge.ingest.pdf_extractor import chunk_text
    from mindforge.ingest.qa_generator import generate_qa_from_chunk, format_qa_as_dpo
    from mindforge.vault.database import Database

    print(f"=== Web URL Ingestion ===")
    print(f"  URL:        {args.url}")
    print(f"  Crawl mode: {'site' if args.crawl else 'single'}")
    if args.crawl:
        print(f"  Max pages:  {args.max_pages}")
        print(f"  Max depth:  {args.max_depth}")
    print()

    # Step 1: Extract content
    if args.crawl:
        print("Crawling site...")
        pages = crawl_site(args.url, max_pages=args.max_pages, max_depth=args.max_depth)
        print(f"  Pages crawled: {len(pages)}")
    else:
        print("Extracting page...")
        page = extract_url(args.url, method="auto")
        pages = [page]
        if page.get("error"):
            print(f"✗ URL extraction failed: {page['error']}")
            print(f"  Tip: Check the URL is accessible. Install beautifulsoup4 if needed: pip install beautifulsoup4 requests")
            return 1
        print(f"  Title: {page.get('title', 'N/A')}")
        print(f"  Method: {page.get('method_used', 'N/A')}")

    if not pages:
        print("✗ No content extracted from the URL.")
        print(f"  Tip: The page may be empty, behind a login wall, or require JavaScript rendering.")
        print(f"  Try a different URL or use --crawl to fetch linked pages too.")
        return 1

    # Step 2: Sanitize all pages
    print("\nSanitizing content (anti-prompt-injection)...")
    all_qa_pairs = []
    all_dpo_entries = []
    total_flags = 0

    for i, page in enumerate(pages):
        content = page.get("content", "")
        if not content:
            continue

        # Sanitize
        san = sanitize_content(content)

        if san["flags"]:
            total_flags += len(san["flags"])
            print(f"  ⚠ Page {i+1}: {len(san['flags'])} injection flag(s) detected")
            for flag in san["flags"]:
                print(f"    - {flag}")

        clean_text = san["clean_text"]
        if not clean_text or len(clean_text) < 50:
            continue

        # Step 3: Chunk and generate Q&A
        chunks = chunk_text(clean_text)
        for chunk in chunks:
            qa_list = generate_qa_from_chunk(chunk, subject=None, adapter=None)
            all_qa_pairs.extend(qa_list)

        # Store web source in database
        try:
            db_path = os.path.join(_project_root, "data", "mindforge.db")
            db = Database(db_path)
            import hashlib
            content_hash = hashlib.sha256(clean_text.encode()).hexdigest()
            db.store_web_source({
                "url": page.get("url", args.url),
                "page_title": page.get("title", ""),
                "content_hash": content_hash,
                "word_count": len(clean_text.split()),
                "extraction_method": page.get("method_used", "beautifulsoup"),
                "sanitization_status": "flagged" if san["flags"] else "clean",
                "injection_flags": json.dumps(san["flags"]) if san["flags"] else None,
                "crawl_mode": "site" if args.crawl else "single",
                "crawl_depth": page.get("depth", 0),
            })
            db.close()
        except Exception as e:
            logging.warning(f"Failed to store web source: {e}")

    print(f"\n  Total Q&A pairs: {len(all_qa_pairs)}")
    print(f"  Total injection flags: {total_flags}")

    # Step 4: Format output
    if all_qa_pairs:
        output_dir = os.path.join(_project_root, "data", "training-data", "dpo")
        os.makedirs(output_dir, exist_ok=True)

        dpo_entries = format_qa_as_dpo(all_qa_pairs)
        output_path = os.path.join(output_dir, "ingest_web.jsonl")
        with open(output_path, "w") as f:
            for entry in dpo_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"\n✓ DPO training data written to: {output_path}")
        print(f"  Total entries: {len(dpo_entries)}")
    else:
        print("\n⚠ No Q&A pairs generated.")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mindforge",
        description="MindForge - AI model probing and correction system",
    )

    # Global flags
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Enable DEBUG-level logging (shows all internal operations)'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true', default=False,
        help='Suppress all output except ERROR-level (useful in scripts)'
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect hardware and available APIs")
    detect_parser.set_defaults(func=cmd_detect)

    # models command (Phase 2)
    models_parser = subparsers.add_parser("models", help="List available models based on hardware and API keys")
    models_parser.set_defaults(func=cmd_models)

    # probe command
    probe_parser = subparsers.add_parser("probe", help="Probe a model against MMLU questions")
    probe_parser.add_argument(
        "--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model name (default: mlx-community/Llama-3.2-3B-Instruct-4bit)"
    )
    probe_parser.add_argument(
        "--subject", default="mathematics",
        help="Subject to probe (default: mathematics)"
    )
    probe_parser.add_argument(
        "--tier", default=1,
        help="Probing tier: 1, 2, 3, or 'all' (default: 1)"
    )
    probe_parser.add_argument(
        "--limit", type=int, default=25,
        help="Number of questions (default: 25)"
    )
    probe_parser.add_argument(
        "--judge-model", default=None,
        help="Model to use as LLM judge (e.g., gpt-4o, openrouter/...)"
    )
    probe_parser.set_defaults(func=cmd_probe)

    # review command
    review_parser = subparsers.add_parser("review", help="Review queued training entries")
    review_parser.add_argument(
        "--limit", type=int, default=100,
        help="Maximum entries to review (default: 100)"
    )
    review_parser.set_defaults(func=cmd_review)

    # format command
    format_parser = subparsers.add_parser("format", help="Format training data")
    format_parser.add_argument(
        "--input", required=True,
        help="Input file path (JSON or JSONL)"
    )
    format_parser.add_argument(
        "--format", default="dpo",
        choices=["dpo", "alpaca", "chatml", "completion", "openai_messages", "template_free"],
        help="Output format (default: dpo)"
    )
    format_parser.add_argument(
        "--output", required=True,
        help="Output file path"
    )
    format_parser.set_defaults(func=cmd_format)

    # convert command (Phase 3)
    convert_parser = subparsers.add_parser("convert", help="Convert a HuggingFace model to MLX format")
    convert_parser.add_argument(
        "--source", required=True,
        help="HuggingFace model repo (e.g., mistralai/Mistral-7B-Instruct-v0.3)"
    )
    convert_parser.add_argument(
        "--quantize", default="4bit",
        help="Quantization: 2bit, 3bit, 4bit, 6bit, 8bit, or 'none'/'full' (default: 4bit)"
    )
    convert_parser.add_argument(
        "--group-size", type=int, default=64,
        help="Quantization group size (default: 64)"
    )
    convert_parser.add_argument(
        "--upload-repo", default=None,
        help="HuggingFace repo to upload the converted model to"
    )
    convert_parser.set_defaults(func=cmd_convert)

    # quantize command (Phase 3)
    quantize_parser = subparsers.add_parser("quantize", help="Re-quantize an existing MLX model")
    quantize_parser.add_argument(
        "--model", required=True,
        help="Path to the existing MLX model directory"
    )
    quantize_parser.add_argument(
        "--bits", type=int, default=4,
        choices=[2, 3, 4, 6, 8],
        help="Target quantization bits (default: 4)"
    )
    quantize_parser.add_argument(
        "--group-size", type=int, default=64,
        help="Quantization group size (default: 64)"
    )
    quantize_parser.add_argument(
        "--upload-repo", default=None,
        help="HuggingFace repo to upload the quantized model to"
    )
    quantize_parser.set_defaults(func=cmd_quantize)

    # train command (Phase 4)
    train_parser = subparsers.add_parser("train", help="Fine-tune a model with SFT, DPO, or ORPO")
    train_parser.add_argument(
        "--model", required=True,
        help="Model name or path (e.g., mlx-community/Llama-3.2-3B-Instruct-4bit)"
    )
    train_parser.add_argument(
        "--data", required=True,
        help="Path to training data directory or file"
    )
    train_parser.add_argument(
        "--mode", default="dpo",
        choices=["sft", "dpo", "orpo"],
        help="Training mode: sft, dpo, or orpo (default: dpo)"
    )
    train_parser.add_argument(
        "--iters", type=int, default=1000,
        help="Number of training iterations (default: 1000)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Training batch size (default: 4)"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    train_parser.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO/ORPO beta parameter (default: 0.1)"
    )
    train_parser.add_argument(
        "--adapter-path", default=None,
        help="Path to save adapter weights (auto-generated if not specified)"
    )
    train_parser.set_defaults(func=cmd_train)

    # evaluate command (Phase 4)
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model on benchmark tasks")
    evaluate_parser.add_argument(
        "--model", required=True,
        help="Model name or path to evaluate"
    )
    evaluate_parser.add_argument(
        "--tasks", default="mmlu_stem",
        help="Evaluation tasks (default: mmlu_stem)"
    )
    evaluate_parser.add_argument(
        "--num-fewshot", type=int, default=5,
        help="Number of few-shot examples (default: 5)"
    )
    evaluate_parser.add_argument(
        "--adapter-path", default=None,
        help="Path to adapter weights for fine-tuned models"
    )
    evaluate_parser.add_argument(
        "--compare", default=None,
        help="Tuned model to compare against (enables comparison mode)"
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    # ingest-pdf command (Phase 5)
    ingest_pdf_parser = subparsers.add_parser("ingest-pdf", help="Ingest a PDF file and generate training data")
    ingest_pdf_parser.add_argument(
        "--file", required=True,
        help="Path to the PDF file"
    )
    ingest_pdf_parser.add_argument(
        "--subject", default=None,
        help="Subject context (e.g., mathematics, physics)"
    )
    ingest_pdf_parser.add_argument(
        "--format", default="dpo",
        choices=["dpo", "json"],
        help="Output format (default: dpo)"
    )
    ingest_pdf_parser.set_defaults(func=cmd_ingest_pdf)

    # ingest-web command (Phase 5)
    ingest_web_parser = subparsers.add_parser("ingest-web", help="Ingest web content from a URL")
    ingest_web_parser.add_argument(
        "--url", required=True,
        help="URL to extract content from"
    )
    ingest_web_parser.add_argument(
        "--crawl", action="store_true", default=False,
        help="Crawl the site (extract linked pages too)"
    )
    ingest_web_parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Maximum pages to crawl (default: 50)"
    )
    ingest_web_parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Maximum crawl depth (default: 3)"
    )
    ingest_web_parser.set_defaults(func=cmd_ingest_web)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nRun 'mindforge detect' to check your hardware and available APIs.")
        print("Run 'mindforge models' to see which models you can use.")
        return 1

    # Set up logging based on --verbose/--quiet flags
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
