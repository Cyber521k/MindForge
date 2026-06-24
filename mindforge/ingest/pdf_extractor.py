"""PDF text extraction and Q&A generation.

Uses pymupdf (fitz) for fast, reliable PDF text extraction.
Text is chunked into manageable sections, then Q&A pairs are generated
from each chunk (via LLM adapter or heuristics).
"""

import os
import hashlib
import logging

logger = logging.getLogger(__name__)


def extract_pdf(file_path):
    """Extract text from a PDF file using pymupdf (fitz).

    Args:
        file_path: Path to the PDF file

    Returns:
        dict with keys:
            - pages: list[str], text content of each page
            - text: str, full text (all pages concatenated)
            - metadata: dict with page_count, word_count, file_hash, filename
    """
    import fitz  # pymupdf

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(file_path)
    pages = []
    full_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        pages.append(page_text)
        full_text_parts.append(page_text)

    doc.close()

    full_text = "\n\n".join(full_text_parts)
    word_count = len(full_text.split())

    # Compute file hash for deduplication
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    metadata = {
        "filename": os.path.basename(file_path),
        "file_path": os.path.abspath(file_path),
        "page_count": len(pages),
        "word_count": word_count,
        "content_hash": hashlib.sha256(full_text.encode()).hexdigest(),
        "file_hash": file_hash,
    }

    return {
        "pages": pages,
        "text": full_text,
        "metadata": metadata,
    }


def chunk_text(text, chunk_size=4000, overlap=200):
    """Split text into overlapping chunks for processing.

    Args:
        text: Full text to chunk
        chunk_size: Maximum characters per chunk (default 4000)
        overlap: Number of characters to overlap between chunks (default 200)

    Returns:
        list[dict]: Each dict has 'text', 'index', 'start_char', 'end_char'
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a sentence or paragraph boundary
        if end < text_len:
            # Look for the last sentence boundary within the last 20% of the chunk
            search_start = start + int(chunk_size * 0.8)
            boundary = -1

            # Look for paragraph break (double newline)
            boundary = text.rfind("\n\n", search_start, end)
            if boundary == -1:
                # Look for single newline
                boundary = text.rfind("\n", search_start, end)
            if boundary == -1:
                # Look for sentence ending
                for ending in ['. ', '! ', '? ']:
                    pos = text.rfind(ending, search_start, end)
                    if pos > boundary:
                        boundary = pos + 1

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "index": len(chunks),
                "start_char": start,
                "end_char": end,
            })

        # Move start position with overlap
        if end >= text_len:
            break
        start = end - overlap if overlap > 0 else end
        # Ensure forward progress
        if start <= chunks[-1]["start_char"]:
            start = chunks[-1]["end_char"]

    return chunks


def generate_qa_pairs(chunks, subject=None, adapter=None):
    """Generate Q&A pairs from a list of text chunks.

    Delegates to qa_generator.generate_qa_from_chunk for each chunk.

    Args:
        chunks: List of chunk dicts (with 'text' key) or strings
        subject: Optional subject context
        adapter: Optional ModelAdapter for LLM-based generation

    Returns:
        list[dict]: Each dict has 'question', 'answer', 'chunk_ref' keys
    """
    from mindforge.ingest.qa_generator import generate_qa_from_chunk

    all_qa = []

    for chunk in chunks:
        chunk_index = chunk.get("index", 0) if isinstance(chunk, dict) else 0

        qa_list = generate_qa_from_chunk(chunk, subject=subject, adapter=adapter)

        for qa in qa_list:
            qa_entry = {
                "question": qa["question"],
                "answer": qa["answer"],
                "chunk_ref": chunk_index,
            }
            all_qa.append(qa_entry)

    return all_qa
