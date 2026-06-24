"""SQLite storage layer for MindForge."""

import os
import sqlite3
import json
import time
import logging

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for storing responses, training entries, and review sessions."""

    def __init__(self, db_path=None):
        """Open (or create) the SQLite database and initialize the schema.

        Args:
            db_path: Path to the SQLite file. Defaults to data/mindforge.db
                     relative to the project root.
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mindforge.db")
        self.db_path = os.path.abspath(db_path)

        # Try to create the data directory; tolerate read-only filesystems
        # (sqlite3.connect will still fail with a clearer error if the dir
        # truly doesn't exist, but at least we won't crash with OSError
        # before giving a useful message).
        data_dir = os.path.dirname(self.db_path)
        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Could not create data directory %s: %s. "
                "Database operations may fail if the file cannot be created.",
                data_dir, e,
            )

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()
        self._ensure_indexes()

    def _init_schema(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()

        # Responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_idx INTEGER,
                prompt TEXT,
                question TEXT,
                choices TEXT,
                correct_answer_idx INTEGER,
                correct_answer_letter TEXT,
                model_response TEXT,
                model_answer_letter TEXT,
                is_correct INTEGER,
                confidence REAL,
                subject TEXT,
                model TEXT,
                created_at REAL
            )
        """)

        # Training entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id INTEGER,
                prompt TEXT,
                chosen TEXT,
                rejected TEXT,
                format TEXT,
                subject TEXT,
                status TEXT DEFAULT 'pending',
                created_at REAL,
                reviewed_at REAL,
                FOREIGN KEY (response_id) REFERENCES responses(id)
            )
        """)

        # Review sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_entry_id INTEGER,
                action TEXT,
                edited_chosen TEXT,
                edited_rejected TEXT,
                reviewed_at REAL,
                FOREIGN KEY (training_entry_id) REFERENCES training_entries(id)
            )
        """)

        # Converted models table (Phase 3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS converted_models (
                id TEXT PRIMARY KEY,
                source_repo TEXT,
                local_path TEXT,
                quantization TEXT,
                model_size_gb REAL,
                uploaded_to_hf BOOLEAN,
                hf_repo TEXT,
                converted_at REAL
            )
        """)

        # Quantized models table (Phase 3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quantized_models (
                id TEXT PRIMARY KEY,
                source_model_id TEXT REFERENCES converted_models(id),
                source_path TEXT,
                output_path TEXT,
                bit_depth INTEGER,
                group_size INTEGER,
                model_size_gb REAL,
                uploaded_to_hf BOOLEAN,
                hf_repo TEXT,
                quantized_at REAL
            )
        """)

        # Training runs table (Phase 4)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT,
                mode TEXT,
                data_path TEXT,
                adapter_path TEXT,
                iters INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                beta REAL,
                status TEXT,
                loss REAL,
                iters_completed INTEGER,
                started_at REAL,
                finished_at REAL
            )
        """)

        # Evaluation results table (Phase 4)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_run_id INTEGER,
                model TEXT,
                task TEXT,
                score REAL,
                metric TEXT,
                details TEXT,
                created_at REAL,
                FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
            )
        """)

        # PDF sources table (Phase 5)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_sources (
                id TEXT PRIMARY KEY,
                filename TEXT,
                file_path TEXT,
                page_count INTEGER,
                word_count INTEGER,
                content_hash TEXT,
                extracted_at TIMESTAMP
            )
        """)

        # Web sources table (Phase 5)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_sources (
                id TEXT PRIMARY KEY,
                url TEXT,
                page_title TEXT,
                content_hash TEXT,
                content_path TEXT,
                word_count INTEGER,
                extraction_method TEXT,
                sanitization_status TEXT,
                injection_flags TEXT,
                crawl_mode TEXT,
                crawl_depth INTEGER,
                crawled_at TIMESTAMP
            )
        """)

        self.conn.commit()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exo_cluster (
                id INTEGER PRIMARY KEY DEFAULT 1,
                running INTEGER,
                installed INTEGER,
                api_url TEXT,
                peer_count INTEGER,
                status TEXT,
                total_memory_gb REAL,
                total_usable_gb REAL,
                devices TEXT,
                rdma_enabled INTEGER,
                updated_at REAL
            )
        """)
        # Ensure the singleton row exists
        cursor.execute("""
            INSERT OR IGNORE INTO exo_cluster (id, running, installed, status, updated_at)
            VALUES (1, 0, 0, 'not_detected', 0)
        """)

        self.conn.commit()

    def _ensure_indexes(self):
        """Create indexes if they don't exist — improves ORDER BY and WHERE filtering."""
        cursor = self.conn.cursor()
        # responses: ordered by created_at DESC, filtered by subject/model
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_responses_subject ON responses(subject)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_responses_model ON responses(model)"
        )
        # training_entries: ordered by created_at, filtered by status
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_entries_created_at ON training_entries(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_entries_status ON training_entries(status)"
        )
        # review_sessions: filtered by training_entry_id
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_sessions_entry_id ON review_sessions(training_entry_id)"
        )
        self.conn.commit()

    def store_response(self, result):
        """Store a probe response in the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO responses (
                question_idx, prompt, question, choices,
                correct_answer_idx, correct_answer_letter,
                model_response, model_answer_letter,
                is_correct, confidence, subject, model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get("question_idx"),
            result.get("prompt"),
            result.get("question"),
            json.dumps(result.get("choices")),
            result.get("correct_answer_idx"),
            result.get("correct_answer_letter"),
            result.get("model_response"),
            result.get("model_answer_letter"),
            1 if result.get("is_correct") else 0,
            result.get("confidence"),
            result.get("subject"),
            result.get("model"),
            time.time(),
        ))
        self.conn.commit()
        result["db_id"] = cursor.lastrowid
        return cursor.lastrowid

    def store_training_entry(self, response_id, prompt, chosen, rejected, format="dpo", subject=None):
        """Store a training entry in the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO training_entries (
                response_id, prompt, chosen, rejected, format, subject, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            response_id, prompt, chosen, rejected, format, subject, time.time(),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_pending_entries(self, limit=100):
        """Get training entries pending review."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM training_entries WHERE status = 'pending' ORDER BY created_at LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_training_entries(self):
        """Get all training entries."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM training_entries ORDER BY created_at")
        return [dict(row) for row in cursor.fetchall()]

    def update_entry_status(self, entry_id, status):
        """Update the status of a training entry."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE training_entries SET status = ?, reviewed_at = ? WHERE id = ?
        """, (status, time.time(), entry_id))
        self.conn.commit()

    def store_review_session(self, training_entry_id, action, edited_chosen=None, edited_rejected=None):
        """Store a review session entry."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO review_sessions (
                training_entry_id, action, edited_chosen, edited_rejected, reviewed_at
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            training_entry_id, action, edited_chosen, edited_rejected, time.time(),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def update_training_entry(self, entry_id, chosen=None, rejected=None):
        """Update the chosen/rejected text of a training entry (for edits)."""
        cursor = self.conn.cursor()
        if chosen is not None:
            cursor.execute("UPDATE training_entries SET chosen = ? WHERE id = ?", (chosen, entry_id))
        if rejected is not None:
            cursor.execute("UPDATE training_entries SET rejected = ? WHERE id = ?", (rejected, entry_id))
        self.conn.commit()

    def store_converted_model(self, model_info):
        """Store a converted model record in the database.

        Args:
            model_info: dict with keys:
                id, source_repo, local_path, quantization,
                model_size_gb, uploaded_to_hf, hf_repo, converted_at

        Returns:
            str: The model ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO converted_models (
                id, source_repo, local_path, quantization,
                model_size_gb, uploaded_to_hf, hf_repo, converted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_info.get("id"),
            model_info.get("source_repo"),
            model_info.get("local_path"),
            model_info.get("quantization"),
            model_info.get("model_size_gb"),
            1 if model_info.get("uploaded_to_hf") else 0,
            model_info.get("hf_repo"),
            model_info.get("converted_at", time.time()),
        ))
        self.conn.commit()
        return model_info.get("id")

    def store_quantized_model(self, model_info):
        """Store a quantized model record in the database.

        Args:
            model_info: dict with keys:
                id, source_path, output_path, bit_depth, group_size,
                model_size_gb, uploaded_to_hf, hf_repo, quantized_at

        Returns:
            str: The model ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO quantized_models (
                id, source_model_id, source_path, output_path,
                bit_depth, group_size, model_size_gb,
                uploaded_to_hf, hf_repo, quantized_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_info.get("id"),
            model_info.get("source_model_id"),
            model_info.get("source_path"),
            model_info.get("output_path"),
            model_info.get("bit_depth"),
            model_info.get("group_size"),
            model_info.get("model_size_gb"),
            1 if model_info.get("uploaded_to_hf") else 0,
            model_info.get("hf_repo"),
            model_info.get("quantized_at", time.time()),
        ))
        self.conn.commit()
        return model_info.get("id")

    def get_converted_models(self):
        """Get all converted model records.

        Returns:
            list of dicts with converted model info
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM converted_models ORDER BY converted_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_quantized_models(self):
        """Get all quantized model records.

        Returns:
            list of dicts with quantized model info
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM quantized_models ORDER BY quantized_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def store_training_run(self, run_info):
        """Store a training run record in the database.

        Args:
            run_info: dict with keys:
                model, mode, data_path, adapter_path, iters, batch_size,
                learning_rate, beta, status, loss, iters_completed,
                started_at, finished_at

        Returns:
            int: The training run ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO training_runs (
                model, mode, data_path, adapter_path, iters, batch_size,
                learning_rate, beta, status, loss, iters_completed,
                started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_info.get("model"),
            run_info.get("mode"),
            run_info.get("data_path"),
            run_info.get("adapter_path"),
            run_info.get("iters"),
            run_info.get("batch_size"),
            run_info.get("learning_rate"),
            run_info.get("beta"),
            run_info.get("status"),
            run_info.get("loss"),
            run_info.get("iters_completed"),
            run_info.get("started_at", time.time()),
            run_info.get("finished_at"),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def update_training_run(self, run_id, updates):
        """Update a training run record.

        Args:
            run_id: The training run ID
            updates: dict of column -> value to update
        """
        cursor = self.conn.cursor()
        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)
        values.append(run_id)
        sql = f"UPDATE training_runs SET {', '.join(set_clauses)} WHERE id = ?"
        cursor.execute(sql, values)
        self.conn.commit()

    def get_training_runs(self, limit=100):
        """Get training run records.

        Args:
            limit: Maximum number of records to return.

        Returns:
            list of dicts with training run info
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def store_evaluation_result(self, eval_info):
        """Store an evaluation result record in the database.

        Args:
            eval_info: dict with keys:
                training_run_id, model, task, score, metric, details

        Returns:
            int: The evaluation result ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO evaluation_results (
                training_run_id, model, task, score, metric, details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_info.get("training_run_id"),
            eval_info.get("model"),
            eval_info.get("task"),
            eval_info.get("score"),
            eval_info.get("metric"),
            eval_info.get("details"),
            time.time(),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_evaluation_results(self, limit=100):
        """Get evaluation result records.

        Args:
            limit: Maximum number of records to return.

        Returns:
            list of dicts with evaluation result info
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM evaluation_results ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def store_exo_status(self, exo_info):
        """Store exo cluster status in the singleton exo_cluster row.

        Args:
            exo_info: dict with keys from detect_exo() and optionally
                      cluster info (total_memory_gb, total_usable_gb,
                      devices, rdma_enabled).

        Returns:
            None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO exo_cluster (
                id, running, installed, api_url, peer_count, status,
                total_memory_gb, total_usable_gb, devices, rdma_enabled,
                updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            1 if exo_info.get("running") else 0,
            1 if exo_info.get("installed") else 0,
            exo_info.get("api_url"),
            exo_info.get("peer_count", 0),
            exo_info.get("status", "not_detected"),
            exo_info.get("total_memory_gb"),
            exo_info.get("total_usable_gb"),
            json.dumps(exo_info.get("devices", [])),
            1 if exo_info.get("rdma_enabled") else 0,
            time.time(),
        ))
        self.conn.commit()

    def get_exo_status(self):
        """Get the exo cluster status from the singleton row.

        Returns:
            dict with exo cluster status, or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM exo_cluster WHERE id = 1")
        row = cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        # Parse devices JSON
        if result.get("devices"):
            try:
                result["devices"] = json.loads(result["devices"])
            except (json.JSONDecodeError, TypeError):
                result["devices"] = []
        else:
            result["devices"] = []
        return result

    def store_pdf_source(self, source_info):
        """Store a PDF source record in the database.

        Args:
            source_info: dict with keys:
                id, filename, file_path, page_count, word_count,
                content_hash, extracted_at

        Returns:
            str: The source ID
        """
        import uuid

        cursor = self.conn.cursor()
        source_id = source_info.get("id") or str(uuid.uuid4())
        cursor.execute("""
            INSERT OR REPLACE INTO pdf_sources (
                id, filename, file_path, page_count, word_count,
                content_hash, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            source_id,
            source_info.get("filename"),
            source_info.get("file_path"),
            source_info.get("page_count"),
            source_info.get("word_count"),
            source_info.get("content_hash"),
            source_info.get("extracted_at", time.time()),
        ))
        self.conn.commit()
        return source_id

    def get_pdf_sources(self):
        """Get all PDF source records.

        Returns:
            list of dicts with PDF source info
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM pdf_sources ORDER BY extracted_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def store_web_source(self, source_info):
        """Store a web source record in the database.

        Args:
            source_info: dict with keys:
                id, url, page_title, content_hash, content_path,
                word_count, extraction_method, sanitization_status,
                injection_flags, crawl_mode, crawl_depth, crawled_at

        Returns:
            str: The source ID
        """
        import uuid

        cursor = self.conn.cursor()
        source_id = source_info.get("id") or str(uuid.uuid4())
        cursor.execute("""
            INSERT OR REPLACE INTO web_sources (
                id, url, page_title, content_hash, content_path,
                word_count, extraction_method, sanitization_status,
                injection_flags, crawl_mode, crawl_depth, crawled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            source_id,
            source_info.get("url"),
            source_info.get("page_title"),
            source_info.get("content_hash"),
            source_info.get("content_path"),
            source_info.get("word_count"),
            source_info.get("extraction_method"),
            source_info.get("sanitization_status"),
            source_info.get("injection_flags"),
            source_info.get("crawl_mode"),
            source_info.get("crawl_depth"),
            source_info.get("crawled_at", time.time()),
        ))
        self.conn.commit()
        return source_id

    def get_web_sources(self):
        """Get all web source records.

        Returns:
            list of dicts with web source info
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM web_sources ORDER BY crawled_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
