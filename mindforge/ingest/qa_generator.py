"""Q&A generation from text chunks.

Two modes:
  - With adapter: Uses an LLM to generate high-quality Q&A pairs
  - Without adapter: Heuristic generation from text structure (sentences → questions)

The generated Q&A pairs can be formatted as DPO entries for training.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)


def generate_qa_from_chunk(chunk, subject=None, adapter=None):
    """Generate Q&A pairs from a text chunk.

    If an adapter (ModelAdapter) is provided, use the LLM to generate
    high-quality Q&A pairs. Otherwise, use heuristic generation from
    text structure.

    Args:
        chunk: Text chunk (str) or dict with 'text' and 'index' keys
        subject: Optional subject context (e.g., 'mathematics')
        adapter: Optional ModelAdapter for LLM-based generation

    Returns:
        list[dict]: Each dict has 'question' and 'answer' keys
    """
    # Normalize chunk input
    if isinstance(chunk, dict):
        text = chunk.get("text", "")
        chunk_index = chunk.get("index", 0)
    else:
        text = chunk
        chunk_index = 0

    if not text or not text.strip():
        return []

    if adapter is not None:
        return _generate_qa_with_llm(text, subject, adapter)
    else:
        return _generate_qa_heuristic(text, subject)


def _generate_qa_with_llm(text, subject, adapter):
    """Use an LLM adapter to generate Q&A pairs from a text chunk.

    Args:
        text: The text chunk
        subject: Optional subject context
        adapter: A ModelAdapter instance

    Returns:
        list[dict]: Q&A pairs with 'question' and 'answer' keys
    """
    subject_context = f" in the domain of {subject}" if subject else ""

    prompt = f"""Read the following text and generate 3-5 question-answer pairs based on its content.
The answers should be factual and grounded in the text{subject_context}.

Text:
---
{text[:4000]}
---

Output the Q&A pairs as a JSON array where each element has "question" and "answer" keys.
Only output the JSON array, no other text.

Example format:
[{{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}]
"""

    try:
        response = adapter.ask(prompt, max_tokens=1024)
        # Try to parse the JSON response
        # Find JSON array in the response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            qa_list = json.loads(json_match.group())
            # Validate structure
            result = []
            for item in qa_list:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    result.append({
                        "question": item["question"],
                        "answer": item["answer"],
                    })
            if result:
                return result
    except Exception as e:
        logger.warning(f"LLM Q&A generation failed: {e}")

    # Fall back to heuristic if LLM fails
    return _generate_qa_heuristic(text, subject)


def _generate_qa_heuristic(text, subject=None):
    """Generate Q&A pairs heuristically from text structure.

    Strategy:
    1. Detect headings/section titles and convert to "What is X?" questions
    2. Extract declarative sentences and convert to questions
    3. Extract definition patterns ("X is Y" → "What is X?")

    Args:
        text: The text chunk
        subject: Optional subject context

    Returns:
        list[dict]: Q&A pairs with 'question' and 'answer' keys
    """
    qa_pairs = []
    lines = text.strip().split("\n")

    # Strategy 1: Heading-based Q&A
    # Lines that are short, don't end with punctuation, and look like titles
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Detect headings: short lines, no ending punctuation, often capitalized
        if (len(line) < 100 and
                not line.endswith(('.', ',', ';', ':', '?', '!')) and
                len(line) > 2):

            # Check if next non-empty line has content (the "answer")
            answer_lines = []
            for j in range(i + 1, min(i + 6, len(lines))):
                next_line = lines[j].strip()
                if next_line:
                    answer_lines.append(next_line)
                if len(answer_lines) >= 3:
                    break

            if answer_lines:
                answer = " ".join(answer_lines)
                # Form question from heading
                question = _heading_to_question(line, subject)
                if question and len(answer) > 20:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer[:500],
                    })

    # Strategy 2: Definition patterns ("X is/are/refers to Y")
    definition_patterns = [
        r'^([A-Z][^.]{2,60}?)\s+(?:is|are|refers to|means|is defined as)\s+(.+?[.!?])',
        r'^([A-Z][^:]{2,60}?):\s*(.+?[.!?])',
    ]

    sentences = _split_sentences(text)
    for sentence in sentences:
        for pattern in definition_patterns:
            match = re.match(pattern, sentence.strip())
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                question = f"What is {term}?"
                if subject:
                    question = f"In {subject}, what is {term}?"
                qa_pairs.append({
                    "question": question,
                    "answer": f"{term} {definition}",
                })
                break  # Only one pattern match per sentence

    # Strategy 3: Convert declarative sentences to questions
    for sentence in sentences[:10]:  # Limit to first 10 sentences
        sentence = sentence.strip()
        if len(sentence) < 30 or len(sentence) > 300:
            continue

        # Skip if already used as a definition
        if any(sentence[:50] in qa.get("answer", "") for qa in qa_pairs):
            continue

        # Simple question transformation
        question = _sentence_to_question(sentence, subject)
        if question:
            qa_pairs.append({
                "question": question,
                "answer": sentence,
            })

    # Deduplicate by question
    seen = set()
    unique_qa = []
    for qa in qa_pairs:
        q_key = qa["question"].lower().strip()
        if q_key not in seen:
            seen.add(q_key)
            unique_qa.append(qa)

    return unique_qa


def _heading_to_question(heading, subject=None):
    """Convert a heading/title to a question.

    Args:
        heading: The heading text
        subject: Optional subject context

    Returns:
        str: A question derived from the heading, or None if no good question
    """
    heading = heading.strip().rstrip(":.")

    # Numbered headings like "1.2.3 Title" → "What is Title?"
    heading = re.sub(r'^[\d.]+\s*', '', heading)

    # Markdown heading markers
    heading = re.sub(r'^#+\s*', '', heading)

    if not heading or len(heading) < 3:
        return None

    subject_prefix = f"In {subject}, " if subject else ""

    # If heading looks like a topic name
    if heading[0].isupper() and len(heading.split()) <= 6:
        return f"{subject_prefix}what is {heading}?"

    return f"{subject_prefix}what does the following describe: \"{heading}\"?"


def _sentence_to_question(sentence, subject=None):
    """Convert a declarative sentence to a question.

    Uses simple transformations:
    - "X is Y." → "What is X?" or "What can you tell me about X?"
    - "X can be used to Y." → "How can X be used?"

    Args:
        sentence: The declarative sentence
        subject: Optional subject context

    Returns:
        str: A question, or None if no good transformation
    """
    sentence = sentence.strip()

    # Pattern: "X is/are Y"
    match = re.match(r'^([A-Z][^.]{2,60}?)\s+(?:is|are)\s+', sentence)
    if match:
        subject_text = match.group(1).strip()
        prefix = f"In {subject}, " if subject else ""
        return f"{prefix}what is {subject_text}?"

    # Pattern: "X can be used to Y"
    match = re.match(r'^([A-Z][^.]{2,60}?)\s+can be used to\s+', sentence, re.IGNORECASE)
    if match:
        subject_text = match.group(1).strip()
        return f"How can {subject_text} be used?"

    # Pattern: "X allows Y" or "X enables Y"
    match = re.match(r'^([A-Z][^.]{2,60}?)\s+(?:allows|enables)\s+', sentence, re.IGNORECASE)
    if match:
        subject_text = match.group(1).strip()
        return f"What does {subject_text} allow or enable?"

    # Generic: ask about the main topic
    # Extract first few words as the topic
    words = sentence.split()
    if len(words) >= 3:
        topic = " ".join(words[:3])
        return f"What can you tell me about: \"{topic}...\"?"

    return None


def _split_sentences(text):
    """Split text into sentences.

    Args:
        text: Input text

    Returns:
        list[str]: List of sentences
    """
    # Simple sentence splitting on common sentence endings
    # Keep the ending punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def format_qa_as_dpo(qa_pairs, prompt_prefix=None):
    """Format Q&A pairs as DPO training entries.

    Each Q&A pair becomes a DPO entry where:
    - prompt = the question (optionally prefixed)
    - chosen = the answer
    - rejected = a generic "I don't know" response

    Args:
        qa_pairs: List of dicts with 'question' and 'answer' keys
        prompt_prefix: Optional prefix to prepend to each prompt

    Returns:
        list[dict]: DPO entries with 'prompt', 'chosen', 'rejected' keys
    """
    dpo_entries = []
    for qa in qa_pairs:
        question = qa.get("question", "")
        answer = qa.get("answer", "")

        if not question or not answer:
            continue

        prompt = f"{prompt_prefix}\n{question}" if prompt_prefix else question

        dpo_entries.append({
            "prompt": prompt,
            "chosen": answer,
            "rejected": "I don't have information about this topic.",
        })

    return dpo_entries
