"""MindForge FastAPI Sidecar — REST + WebSocket server.

Wraps the mindforge CLI / package as API endpoints for the Tauri desktop app.
Run: python3 server.py  (listens on localhost:7878)
"""

import os
import sys
import json
import time
import uuid
import asyncio
import threading
import logging
from contextlib import asynccontextmanager, suppress
from typing import Optional

# Ensure project root is on sys.path so we can import mindforge
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mindforge.sidecar")

# ---------------------------------------------------------------------------
# Event loop reference for cross-thread WebSocket broadcasting
# ---------------------------------------------------------------------------

_main_loop: asyncio.AbstractEventLoop | None = None
_ws_clients: list[WebSocket] = []


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop (set during startup)."""
    if _main_loop is None:
        raise RuntimeError("Event loop not initialised yet")
    return _main_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler — stores the event loop on startup, cleans up on shutdown."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("MindForge sidecar starting up")
    yield
    # Shutdown: close all WebSocket clients
    for ws in list(_ws_clients):
        try:
            await ws.close()
        except Exception:
            pass
    _ws_clients.clear()
    logger.info("MindForge sidecar shut down")


app = FastAPI(title="MindForge Sidecar", version="0.0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:1420",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:1420",
        "http://127.0.0.1:3000",
        "http://localhost:7878",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Job tracking + WebSocket broadcast
# ---------------------------------------------------------------------------

_jobs: dict = {}


# ---------------------------------------------------------------------------
# Hardware cache (avoid running sysctl subprocesses on every /api/hardware call)
# ---------------------------------------------------------------------------

_hardware_cache: dict | None = None


def public_job(job: dict) -> dict:
    """Return a JSON-serializable view of a job."""
    return {
        key: value
        for key, value in job.items()
        if key not in {"cancel_event", "thread"}
    }


async def broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients (async)."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        try:
            await ws.close()
        except Exception:
            pass


def emit(message: dict):
    """Thread-safe broadcast — callable from background threads."""
    coro = broadcast(message)
    try:
        asyncio.run_coroutine_threadsafe(coro, get_loop())
    except RuntimeError:
        # Loop not ready yet — close the orphaned coroutine to avoid
        # "coroutine was never awaited" RuntimeWarning.
        coro.close()


def create_job(job_type: str) -> str:
    """Create a new job entry and return its 8-char ID."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "running",
        "type": job_type,
        "result": None,
        "error": None,
        "progress": 0,
        "started_at": time.time(),
        "cancel_event": threading.Event(),
        "thread": None,
    }
    return job_id


def start_job_thread(job_id: str, target):
    """Start a daemon thread for a job and store the thread reference."""
    thread = threading.Thread(target=target, daemon=True)
    if job_id in _jobs:
        _jobs[job_id]["thread"] = thread
    thread.start()
    return thread


def is_job_cancelled(job_id: str) -> bool:
    """Check whether a job has been cancelled via cancel_job()."""
    return job_id in _jobs and _jobs[job_id]["cancel_event"].is_set()


def cancel_job(job_id: str):
    """Set a job's cancel event and emit a job_cancelled WebSocket message."""
    if job_id in _jobs:
        _jobs[job_id]["cancel_event"].set()
        _jobs[job_id]["status"] = "cancelled"
        emit({"type": "job_cancelled", "job_id": job_id})


def finish_job(job_id: str, result: dict):
    """Mark a job as completed, store its result, and emit job_complete."""
    if job_id in _jobs and _jobs[job_id]["status"] != "cancelled":
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result
        _jobs[job_id]["progress"] = 100
        emit({"type": "job_complete", "job_id": job_id, "job_type": _jobs[job_id]["type"], "result": result})


def fail_job(job_id: str, error: str):
    """Mark a job as failed, store the error, and emit job_failed."""
    if job_id in _jobs and _jobs[job_id]["status"] != "cancelled":
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = error
        emit({"type": "job_failed", "job_id": job_id, "error": error})


def update_progress(job_id: str, progress: float, **extra):
    """Update job progress and emit a WebSocket progress message."""
    if job_id in _jobs:
        _jobs[job_id]["progress"] = progress
        msg = {"type": "progress", "job_id": job_id, "progress": progress, **extra}
        emit(msg)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class ProbeRequest(BaseModel):
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    subject: str = "mathematics"
    tier: str = "1"
    limit: int = 25
    judge_model: Optional[str] = None


class ReviewRequest(BaseModel):
    action: str
    edited_chosen: Optional[str] = None
    edited_rejected: Optional[str] = None


class FormatRequest(BaseModel):
    input: str
    format: str = "dpo"
    output: str = ""


class ConvertRequest(BaseModel):
    source: str
    quantize: str = "4bit"
    group_size: int = 64
    upload_repo: Optional[str] = None


class QuantizeRequest(BaseModel):
    model: str
    bits: int = 4
    group_size: int = 64
    upload_repo: Optional[str] = None


class TrainRequest(BaseModel):
    model: str
    data: str
    mode: str = "dpo"
    iters: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-5
    beta: float = 0.1
    adapter_path: Optional[str] = None


class EvaluateRequest(BaseModel):
    model: str
    tasks: str = "mmlu_stem"
    num_fewshot: int = 5
    adapter_path: Optional[str] = None


class IngestPdfRequest(BaseModel):
    file: str
    subject: Optional[str] = None
    format: str = "dpo"


class IngestWebRequest(BaseModel):
    url: str
    crawl: bool = False
    max_pages: int = 50
    max_depth: int = 3


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/hardware")
async def get_hardware():
    """Hardware detection — wraps mindforge.hardware.detector (cached per session)."""
    global _hardware_cache
    if _hardware_cache is not None:
        return _hardware_cache
    try:
        from mindforge.hardware.detector import detect_hardware

        _hardware_cache = detect_hardware()
        return _hardware_cache
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hardware detection failed: {e}")


@app.get("/api/models")
async def get_models():
    """Available models — wraps mindforge.hardware.model_list."""
    try:
        from mindforge.hardware.model_list import get_available_models

        models = get_available_models()
        return models
    except ImportError:
        try:
            from mindforge.hardware.detector import detect_hardware

            hw = detect_hardware()
            return {"hardware": hw, "local": [], "cloud": []}
        except Exception as e:
            logger.error(f"Model list fallback failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model list failed: {e}")
    except Exception as e:
        logger.error(f"Model list failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model list failed: {e}")


@app.get("/api/taxonomy")
async def get_taxonomy():
    """Subject catalog — reads taxonomy/subjects.yaml."""
    tax_path = os.path.join(_PROJECT_ROOT, "taxonomy", "subjects.yaml")

    def _load():
        """Load the taxonomy YAML from disk (runs in thread pool)."""
        import yaml

        with open(tax_path, "r") as f:
            return yaml.safe_load(f)

    try:
        return await asyncio.to_thread(_load)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Taxonomy file not found: {tax_path}")
    except Exception as e:
        logger.error(f"Taxonomy load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Taxonomy load failed: {e}")


@app.get("/api/responses")
async def get_responses():
    """Get probe results from the SQLite database."""
    try:
        def _query():
            """Query recent responses from the SQLite database (runs in thread pool)."""
            from mindforge.vault.database import Database

            db = Database(os.path.join(_PROJECT_ROOT, "data", "mindforge.db"))
            try:
                cursor = db.conn.cursor()
                cursor.execute("SELECT * FROM responses ORDER BY created_at DESC LIMIT 100")
                rows = [dict(r) for r in cursor.fetchall()]
                return rows
            finally:
                db.close()

        return await asyncio.to_thread(_query)
    except Exception as e:
        logger.error(f"Failed to get responses: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get responses: {e}")


@app.get("/api/training-entries")
async def get_training_entries():
    """Get all training entries from the database."""
    try:
        def _query():
            """Query all training entries from the SQLite database (runs in thread pool)."""
            from mindforge.vault.database import Database

            db = Database(os.path.join(_PROJECT_ROOT, "data", "mindforge.db"))
            try:
                entries = db.get_all_training_entries()
                return entries
            finally:
                db.close()

        return await asyncio.to_thread(_query)
    except Exception as e:
        logger.error(f"Failed to get training entries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training entries: {e}")


@app.post("/api/probe")
async def start_probe(req: ProbeRequest):
    """Start a probing job — returns job_id, streams progress via WebSocket."""
    if not req.model:
        raise HTTPException(status_code=400, detail="Model is required")

    job_id = create_job("probe")

    def run_probe():
        """Thread target: run the probe engine and emit progress via WebSocket."""
        try:
            if is_job_cancelled(job_id):
                return
            from mindforge.probe.engine import ProbeEngine

            emit({"type": "probe_started", "job_id": job_id, "model": req.model, "subject": req.subject})
            engine = ProbeEngine(
                model_name=req.model,
                subject=req.subject,
                tier=int(req.tier) if req.tier != "all" else 1,
                limit=req.limit,
            )
            result = engine.run()
            finish_job(job_id, result)
        except Exception as e:
            fail_job(job_id, str(e))
            logger.error(f"Probe job {job_id} failed: {e}")

    start_job_thread(job_id, run_probe)
    return {"job_id": job_id, "status": "started"}


@app.get("/api/probe/{job_id}")
async def get_probe_status(job_id: str):
    """Get the status of a probing job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return public_job(_jobs[job_id])


@app.post("/api/review/{entry_id}")
async def submit_review(entry_id: int, req: ReviewRequest):
    """Submit a review action (accept / reject / edit / skip)."""
    valid_actions = {"accept", "reject", "edit", "skip"}
    if req.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{req.action}'. Must be one of: {', '.join(sorted(valid_actions))}",
        )

    def _review():
        """Submit the review action to the database (runs in thread pool)."""
        from mindforge.vault.database import Database

        db = Database(os.path.join(_PROJECT_ROOT, "data", "mindforge.db"))
        try:
            cursor = db.conn.cursor()
            cursor.execute("SELECT id FROM training_entries WHERE id = ?", (entry_id,))
            if cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"Training entry {entry_id} not found")

            if req.action == "accept":
                db.update_entry_status(entry_id, "accepted")
            elif req.action == "reject":
                db.update_entry_status(entry_id, "rejected")
            elif req.action == "edit":
                db.update_training_entry(entry_id, chosen=req.edited_chosen, rejected=req.edited_rejected)
                db.update_entry_status(entry_id, "edited")
            elif req.action == "skip":
                pass  # Leave as pending

            db.store_review_session(entry_id, req.action, req.edited_chosen, req.edited_rejected)
            return {"status": "ok", "action": req.action}
        finally:
            db.close()

    try:
        result = await asyncio.to_thread(_review)
        emit({"type": "review_action", "entry_id": entry_id, "action": req.action})
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Review action failed: {e}")
        raise HTTPException(status_code=500, detail=f"Review action failed: {e}")


@app.post("/api/format")
async def format_data(req: FormatRequest):
    """Format training data into the requested format (DPO default)."""
    import json as _json

    def _format():
        """Load and format training data from file (runs in thread pool)."""
        try:
            with open(req.input, "r") as f:
                if req.input.endswith(".jsonl"):
                    entries = [_json.loads(line) for line in f if line.strip()]
                else:
                    entries = _json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Input file not found: {req.input}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in input file: {e}")

        fmt = req.format
        if fmt == "dpo":
            from mindforge.format.dpo import format_dpo_batch, write_dpo_jsonl

            formatted = format_dpo_batch(entries)
            out = req.output or os.path.join(_PROJECT_ROOT, "data", "training-data", "dpo", "train.jsonl")
            write_dpo_jsonl(formatted, out)
        elif fmt == "alpaca":
            from mindforge.format.alpaca import format_alpaca_batch

            formatted = format_alpaca_batch(entries)
            out = req.output or "output.json"
            with open(out, "w") as f:
                _json.dump(formatted, f, indent=2)
        elif fmt == "chatml":
            from mindforge.format.chatml import format_chatml_batch

            formatted = format_chatml_batch(entries)
            out = req.output or "output.jsonl"
            with open(out, "w") as f:
                for e in formatted:
                    f.write(_json.dumps(e) + "\n")
        elif fmt == "completion":
            from mindforge.format.completion import format_completion_batch

            formatted = format_completion_batch(entries)
            out = req.output or "output.jsonl"
            with open(out, "w") as f:
                for e in formatted:
                    f.write(_json.dumps(e) + "\n")
        elif fmt == "openai_messages":
            from mindforge.format.openai_messages import format_openai_messages_batch

            formatted = format_openai_messages_batch(entries)
            out = req.output or "output.jsonl"
            with open(out, "w") as f:
                for e in formatted:
                    f.write(_json.dumps(e) + "\n")
        elif fmt == "template_free":
            from mindforge.format.template_free import format_template_free_batch

            formatted = format_template_free_batch(entries)
            out = req.output or "output.jsonl"
            with open(out, "w") as f:
                for e in formatted:
                    f.write(_json.dumps(e) + "\n")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")

        return {"status": "ok", "count": len(formatted), "output": out}

    try:
        return await asyncio.to_thread(_format)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Format failed: {e}")
        raise HTTPException(status_code=500, detail=f"Format failed: {e}")


@app.post("/api/convert")
async def convert_model_api(req: ConvertRequest):
    """Convert a model to MLX format (background job)."""
    if not req.source:
        raise HTTPException(status_code=400, detail="Source model is required")

    job_id = create_job("convert")

    def run():
        """Thread target: execute the job and emit progress/fail/complete via WebSocket."""
        try:
            if is_job_cancelled(job_id):
                return
            from mindforge.convert.converter import convert_model

            q_bits_map = {
                "2bit": 2, "3bit": 3, "4bit": 4, "6bit": 6, "8bit": 8,
                "none": None, "full": None,
            }
            q_bits = q_bits_map.get(req.quantize, 4)
            result = convert_model(
                req.source,
                quantize=q_bits is not None,
                q_bits=q_bits,
                q_group_size=req.group_size,
                upload_repo=req.upload_repo,
            )
            finish_job(job_id, result)
        except Exception as e:
            fail_job(job_id, str(e))
            logger.error(f"Convert job {job_id} failed: {e}")

    start_job_thread(job_id, run)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/quantize")
async def quantize_model_api(req: QuantizeRequest):
    """Quantize a model (background job)."""
    if not req.model:
        raise HTTPException(status_code=400, detail="Model path is required")

    job_id = create_job("quantize")

    def run():
        """Thread target: execute the job and emit progress/fail/complete via WebSocket."""
        try:
            if is_job_cancelled(job_id):
                return
            from mindforge.convert.quantizer import quantize_model

            result = quantize_model(
                req.model, bits=req.bits, group_size=req.group_size, upload_repo=req.upload_repo
            )
            finish_job(job_id, result)
        except Exception as e:
            fail_job(job_id, str(e))
            logger.error(f"Quantize job {job_id} failed: {e}")

    start_job_thread(job_id, run)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/train")
async def start_training(req: TrainRequest):
    """Start a training job — streams progress (loss, iteration) via WebSocket."""
    if not req.model:
        raise HTTPException(status_code=400, detail="Model is required")
    if not req.data:
        raise HTTPException(status_code=400, detail="Data path is required")

    job_id = create_job("train")

    def run():
        """Thread target: execute the job and emit progress/fail/complete via WebSocket."""
        try:
            if is_job_cancelled(job_id):
                return
            emit({"type": "train_started", "job_id": job_id, "model": req.model, "mode": req.mode})
            from mindforge.train.trainer import train_model

            result = train_model(
                req.model,
                req.data,
                mode=req.mode,
                iters=req.iters,
                batch_size=req.batch_size,
                learning_rate=req.learning_rate,
                beta=req.beta,
                adapter_path=req.adapter_path,
            )
            finish_job(job_id, result)
        except Exception as e:
            fail_job(job_id, str(e))
            logger.error(f"Train job {job_id} failed: {e}")

    start_job_thread(job_id, run)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/evaluate")
async def start_evaluation(req: EvaluateRequest):
    """Start an evaluation job — streams progress via WebSocket."""
    if not req.model:
        raise HTTPException(status_code=400, detail="Model is required")

    job_id = create_job("evaluate")

    def run():
        """Thread target: execute the job and emit progress/fail/complete via WebSocket."""
        try:
            if is_job_cancelled(job_id):
                return
            emit({"type": "eval_started", "job_id": job_id, "model": req.model})
            from mindforge.evaluate.evaluator import evaluate_model

            result = evaluate_model(
                req.model,
                tasks=req.tasks,
                num_fewshot=req.num_fewshot,
                adapter_path=req.adapter_path,
            )
            finish_job(job_id, result)
        except Exception as e:
            fail_job(job_id, str(e))
            logger.error(f"Evaluate job {job_id} failed: {e}")

    start_job_thread(job_id, run)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/ingest-pdf")
async def ingest_pdf(req: IngestPdfRequest):
    """Ingest a PDF — extracts text, generates Q&A pairs, formats as DPO."""
    if not req.file:
        raise HTTPException(status_code=400, detail="File path is required")

    job_id = create_job("ingest-pdf")

    def run():
        """Thread target: execute the job and emit progress/fail/complete via WebSocket."""
        try:
            if is_job_cancelled(job_id):
                return
            from mindforge.ingest.pdf_extractor import extract_pdf, chunk_text, generate_qa_pairs
            from mindforge.ingest.qa_generator import format_qa_as_dpo
            from mindforge.vault.database import Database

            emit({"type": "ingest_started", "job_id": job_id, "source": req.file})

            if not os.path.exists(req.file):
                fail_job(job_id, f"File not found: {req.file}")
                return

            # Step 1: Extract text
            pdf_data = extract_pdf(req.file)
            if is_job_cancelled(job_id):
                return
            emit({"type": "progress", "job_id": job_id, "progress": 25, "stage": "extracted"})

            # Step 2: Chunk text
            chunks = chunk_text(pdf_data["text"])
            if is_job_cancelled(job_id):
                return
            emit({"type": "progress", "job_id": job_id, "progress": 50, "stage": "chunked", "chunks": len(chunks)})

            # Step 3: Generate Q&A pairs
            qa_pairs = generate_qa_pairs(chunks, subject=req.subject, adapter=None)
            if is_job_cancelled(job_id):
                return
            emit({"type": "progress", "job_id": job_id, "progress": 75, "stage": "qa_generated", "qa_pairs": len(qa_pairs)})

            # Step 4: Format as DPO and write output
            dpo_entries = format_qa_as_dpo(qa_pairs)
            output_dir = os.path.join(_PROJECT_ROOT, "data", "training-data", "dpo")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "ingest_pdf.jsonl")
            with open(output_path, "w") as f:
                for entry in dpo_entries:
                    f.write(json.dumps(entry) + "\n")

            # Step 5: Store source in database
            try:
                db_path = os.path.join(_PROJECT_ROOT, "data", "mindforge.db")
                db = Database(db_path)
                try:
                    db.store_pdf_source({
                        "filename": pdf_data["metadata"]["filename"],
                        "file_path": pdf_data["metadata"]["file_path"],
                        "page_count": pdf_data["metadata"]["page_count"],
                        "word_count": pdf_data["metadata"]["word_count"],
                        "content_hash": pdf_data["metadata"]["content_hash"],
                    })
                finally:
                    db.close()
            except Exception as db_err:
                logger.warning(f"Failed to store PDF source in database: {db_err}")

            finish_job(job_id, {
                "qa_pairs": len(qa_pairs),
                "dpo_entries": len(dpo_entries),
                "output_path": output_path,
            })
        except Exception as e:
            fail_job(job_id, str(e))
            logger.error(f"PDF ingest job {job_id} failed: {e}")

    start_job_thread(job_id, run)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/ingest-web")
async def ingest_web(req: IngestWebRequest):
    """Ingest a web URL — extracts content, sanitizes, generates Q&A, formats as DPO."""
    if not req.url:
        raise HTTPException(status_code=400, detail="URL is required")

    job_id = create_job("ingest-web")

    def run():
        """Thread target: execute the job and emit progress/fail/complete via WebSocket."""
        try:
            if is_job_cancelled(job_id):
                return
            from mindforge.ingest.web_extractor import extract_url, crawl_site
            from mindforge.ingest.sanitizer import sanitize_content
            from mindforge.ingest.pdf_extractor import chunk_text
            from mindforge.ingest.qa_generator import generate_qa_from_chunk, format_qa_as_dpo
            from mindforge.vault.database import Database

            emit({"type": "ingest_started", "job_id": job_id, "source": req.url})

            # Step 1: Extract content
            if req.crawl:
                pages = crawl_site(req.url, max_pages=req.max_pages, max_depth=req.max_depth)
            else:
                page = extract_url(req.url, method="auto")
                if page.get("error"):
                    fail_job(job_id, f"Extraction failed: {page['error']}")
                    return
                pages = [page]

            if is_job_cancelled(job_id):
                return
            if not pages:
                fail_job(job_id, "No content extracted")
                return

            emit({"type": "progress", "job_id": job_id, "progress": 25, "stage": "extracted", "pages": len(pages)})

            # Step 2: Sanitize, chunk, and generate Q&A for each page
            all_qa_pairs = []
            total_flags = 0

            # Open DB connection once for all pages (was N+1: one connection per page)
            db_path = os.path.join(_PROJECT_ROOT, "data", "mindforge.db")
            db = Database(db_path)
            try:
                for i, page in enumerate(pages):
                    if is_job_cancelled(job_id):
                        return
                    content = page.get("content", "")
                    if not content:
                        continue

                    san = sanitize_content(content)
                    if san["flags"]:
                        total_flags += len(san["flags"])
                        logger.warning(f"Page {i+1}: {len(san['flags'])} injection flag(s) detected")

                    clean_text = san["clean_text"]
                    if not clean_text or len(clean_text) < 50:
                        continue

                    chunks = chunk_text(clean_text)
                    for chunk in chunks:
                        if is_job_cancelled(job_id):
                            return
                        qa_list = generate_qa_from_chunk(chunk, subject=None, adapter=None)
                        all_qa_pairs.extend(qa_list)

                    # Store web source in database (reuse existing connection)
                    try:
                        import hashlib
                        content_hash = hashlib.sha256(clean_text.encode()).hexdigest()
                        db.store_web_source({
                            "url": page.get("url", req.url),
                            "page_title": page.get("title", ""),
                            "content_hash": content_hash,
                            "word_count": len(clean_text.split()),
                            "extraction_method": page.get("method_used", "beautifulsoup"),
                            "sanitization_status": "flagged" if san["flags"] else "clean",
                            "injection_flags": json.dumps(san["flags"]) if san["flags"] else None,
                            "crawl_mode": "site" if req.crawl else "single",
                            "crawl_depth": page.get("depth", 0),
                        })
                    except Exception as db_err:
                        logger.warning(f"Failed to store web source: {db_err}")
            finally:
                db.close()

            emit({"type": "progress", "job_id": job_id, "progress": 75, "stage": "qa_generated", "qa_pairs": len(all_qa_pairs)})

            # Step 3: Format as DPO and write output
            dpo_entries = format_qa_as_dpo(all_qa_pairs)
            output_dir = os.path.join(_PROJECT_ROOT, "data", "training-data", "dpo")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "ingest_web.jsonl")
            with open(output_path, "w") as f:
                for entry in dpo_entries:
                    f.write(json.dumps(entry) + "\n")

            finish_job(job_id, {
                "pages": len(pages),
                "qa_pairs": len(all_qa_pairs),
                "dpo_entries": len(dpo_entries),
                "output_path": output_path,
                "injection_flags": total_flags,
            })
        except Exception as e:
            fail_job(job_id, str(e))
            logger.error(f"Web ingest job {job_id} failed: {e}")

    start_job_thread(job_id, run)
    return {"job_id": job_id, "status": "started"}


@app.get("/api/stats")
async def get_stats():
    """Get aggregate statistics from the database."""
    def _stats():
        """Compute aggregate statistics from the SQLite database (runs in thread pool)."""
        from mindforge.vault.database import Database

        db = Database(os.path.join(_PROJECT_ROOT, "data", "mindforge.db"))
        try:
            cursor = db.conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM responses")
            total_q = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM training_entries")
            training_pairs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT subject) FROM responses")
            subjects = cursor.fetchone()[0]

            cursor.execute("SELECT subject, AVG(is_correct) * 100 as accuracy FROM responses GROUP BY subject")
            accuracy = {}
            for row in cursor.fetchall():
                domain = row[0].split("_")[0] if row[0] else "unknown"
                if domain not in accuracy:
                    accuracy[domain] = []
                accuracy[domain].append(row[1])
            accuracy = {k: sum(v) / len(v) for k, v in accuracy.items()}

            cursor.execute("SELECT COUNT(*) FROM training_runs")
            training_runs = cursor.fetchone()[0]

            return {
                "total_questions": total_q,
                "training_pairs": training_pairs,
                "subjects": subjects,
                "training_runs": training_runs,
                "accuracy": accuracy,
            }
        finally:
            db.close()

    try:
        return await asyncio.to_thread(_stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs (probe, train, evaluate, etc.)."""
    return {job_id: public_job(job) for job_id, job in _jobs.items()}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get the status of a job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return public_job(_jobs[job_id])


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_api(job_id: str):
    """Cancel a running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if _jobs[job_id]["status"] != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    cancel_job(job_id)
    return {"status": "cancelled", "job_id": job_id}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint — broadcasts progress updates to connected clients.

    Features:
    - On connect: sends current job statuses for reconnection recovery
    - Heartbeat: sends {type: 'heartbeat'} every 30 seconds
    - Error boundary: if any send fails, the connection is cleaned up gracefully
    - Clean disconnect: heartbeat cancelled, WS removed from _ws_clients, closed
    """
    await ws.accept()
    _ws_clients.append(ws)

    # Send current job statuses on connect (for reconnection recovery)
    try:
        await ws.send_json({
            "type": "job_statuses",
            "jobs": {job_id: public_job(job) for job_id, job in _jobs.items()},
        })
    except Exception:
        # Client disconnected immediately after connect — clean up
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        try:
            await ws.close()
        except Exception:
            pass
        return

    async def heartbeat():
        """Send periodic WebSocket heartbeats every 30 seconds to keep the connection alive."""
        while True:
            await asyncio.sleep(30)
            try:
                await ws.send_json({"type": "heartbeat", "timestamp": time.time()})
            except Exception:
                # Connection lost — stop heartbeating
                break

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
            elif data.startswith("{"):
                # Allow client to send JSON commands
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "subscribe":
                        await ws.send_json({"type": "subscribed", "channels": msg.get("channels", ["*"])})
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=7878, log_level="info")
