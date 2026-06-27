"""Automated review system using LLM-as-judge and web search.

The AutoReviewer evaluates training entries (prompt/chosen/rejected pairs) using
a judge LLM. If the judge is uncertain (below the configured web search
threshold), it falls back to web search to find the correct answer. This
automates the review queue that previously required manual human sign-off.
"""

import os
import re
import json
import logging
import hashlib
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class _LinkExtractor(HTMLParser):
    """Small stdlib fallback for extracting links when BeautifulSoup is absent."""

    def __init__(self, base_url):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href")
        if href:
            self._href = urljoin(self.base_url, href)
            self._text = []

    def handle_data(self, data):
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            self.links.append({
                "url": self._href,
                "text": " ".join(self._text).strip(),
                "snippet": "",
            })
            self._href = None
            self._text = []


class _ReadableTextExtractor(HTMLParser):
    """Extract paragraph-like page content without external dependencies."""

    CAPTURE_TAGS = {"p", "li", "blockquote", "h1", "h2", "h3"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "form"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fragments = []
        self._skip_depth = 0
        self._capture_tag = None
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self.CAPTURE_TAGS:
            self._capture_tag = tag
            self._buffer = []

    def handle_data(self, data):
        if self._skip_depth == 0 and self._capture_tag:
            self._buffer.append(data)

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag == self._capture_tag:
            text = AutoReviewer._clean_text(" ".join(self._buffer))
            if text:
                self.fragments.append(text)
            self._capture_tag = None
            self._buffer = []


class AutoReviewer:
    """Automated review of training entries using LLM-as-judge + web search.

    Workflow per entry:
        1. Send question + model answer + chosen/rejected to judge LLM
        2. Judge evaluates correctness of chosen and rejected
        3. If judge confidence is below the configured threshold, search the web
        4. If web search finds a better answer, correct the chosen/rejected pair
        5. Return action: accept / reject / edit
    """

    # Confidence threshold below which web search is triggered
    WEB_SEARCH_THRESHOLD = 0.7

    LEGACY_RESULT_KEYS = (
        "action",
        "confidence",
        "explanation",
        "edited_chosen",
        "edited_rejected",
        "web_source",
    )

    def __init__(self, judge_adapter=None, web_search_enabled=True, web_search_threshold=None):
        """Initialize the auto-reviewer.

        Args:
            judge_adapter: A ModelAdapter for the reviewing LLM (OpenAIAdapter,
                          OpenRouterAdapter, etc.). If None, auto-detects from
                          available API keys or falls back to Ollama/MLX.
            web_search_enabled: Whether to use web search when the judge is uncertain.
            web_search_threshold: Confidence threshold below which web search is triggered.
        """
        self.judge_adapter = judge_adapter
        self.web_search_enabled = web_search_enabled
        if web_search_threshold is None:
            web_search_threshold = self.WEB_SEARCH_THRESHOLD
        self.web_search_threshold = float(web_search_threshold)
        self.judge_model_name = "auto"

        if self.judge_adapter is None:
            self.judge_adapter = self._auto_detect_judge()

        if self.judge_adapter:
            self.judge_model_name = getattr(self.judge_adapter, "model_name", "unknown")
            logger.info(f"AutoReviewer using judge model: {self.judge_model_name}")
        else:
            logger.warning("No judge adapter available. Reviews will be limited.")

    def _auto_detect_judge(self):
        """Auto-detect an available judge model from API keys or local models.

        Priority:
            1. OpenAI (OPENAI_API_KEY) — GPT-4o
            2. OpenRouter (OPENROUTER_API_KEY) — Claude/mixtral
            3. Ollama (if running locally)
            4. MLX local model (if available)
        """
        from mindforge.probe.adapters import create_adapter

        # 1. OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            try:
                adapter = create_adapter("gpt-4o")
                logger.info("Auto-detected OpenAI judge (gpt-4o)")
                return adapter
            except Exception as e:
                logger.debug(f"OpenAI adapter creation failed: {e}")

        # 2. OpenRouter
        if os.environ.get("OPENROUTER_API_KEY"):
            try:
                adapter = create_adapter("openrouter/anthropic/claude-3.5-sonnet")
                logger.info("Auto-detected OpenRouter judge (claude-3.5-sonnet)")
                return adapter
            except Exception as e:
                logger.debug(f"OpenRouter adapter creation failed: {e}")

        # 3. Ollama (check if running locally)
        try:
            from mindforge.hardware.ollama_detector import detect_ollama
            ollama_info = detect_ollama()
            if ollama_info.get("running") and ollama_info.get("models"):
                # detect_ollama() returns model names as strings, not dicts
                model_name = ollama_info["models"][0]
                if isinstance(model_name, dict):
                    model_name = model_name.get("name", "llama3.1:latest")
                adapter = create_adapter(f"ollama/{model_name}")
                logger.info(f"Auto-detected Ollama judge ({model_name})")
                return adapter
        except Exception as e:
            logger.debug(f"Ollama detection failed: {e}")

        # 4. MLX local model fallback
        try:
            adapter = create_adapter("mlx-community/Llama-3.2-3B-Instruct-4bit")
            logger.info("Auto-detected MLX judge (Llama-3.2-3B-Instruct-4bit)")
            return adapter
        except Exception as e:
            logger.debug(f"MLX adapter creation failed: {e}")

        return None

    def review_entry(self, entry):
        """Review a single training entry.

        Args:
            entry: dict with keys: prompt, chosen, rejected, subject,
                   question, model_answer, correct_answer

        Returns:
            dict with keys:
                - action: "accept" / "reject" / "edit"
                - confidence: float 0.0-1.0
                - explanation: str
                - edited_chosen: str or None
                - edited_rejected: str or None
                - web_source: dict or None
        """
        detailed = self.review_entry_detailed(entry)
        return {key: detailed.get(key) for key in self.LEGACY_RESULT_KEYS}

    def review_entry_detailed(self, entry):
        """Review a single training entry and include detailed judge diagnostics.

        Returns all fields from review_entry plus:
            - reasoning: combined judge reasoning for chosen and rejected answers
            - chosen_verdict: independent verdict for the chosen answer
            - rejected_verdict: independent verdict for the rejected answer
            - cross_check_passed: False when rejected is judged right and chosen wrong
        """
        question = entry.get("question") or entry.get("prompt", "")
        chosen = entry.get("chosen", "")
        rejected = entry.get("rejected", "")
        claimed_correct = entry.get("correct_answer")

        # Step 1: Judge chosen and rejected independently, then cross-check them.
        cross_check = self._cross_verify(question, chosen, rejected, claimed_correct)
        chosen_verdict = cross_check["chosen_verdict"]
        rejected_verdict = cross_check["rejected_verdict"]
        confidence = chosen_verdict.get("confidence", 0.5)
        chosen_correct = chosen_verdict.get("correct", True)
        explanation = chosen_verdict.get("explanation", "")
        reasoning = cross_check.get("reasoning", "")

        web_source = None
        edited_chosen = None
        edited_rejected = None

        # Step 2: If the rejected answer is independently better, swap the pair.
        if cross_check.get("should_edit"):
            edited_chosen = rejected
            edited_rejected = chosen
            confidence = max(
                confidence,
                rejected_verdict.get("confidence", 0.5),
            )
            explanation = (
                f"{explanation} Cross-check found the rejected answer is correct "
                "while the chosen answer is incorrect."
            ).strip()

        # Step 3: If uncertain and not already corrected, try web search.
        if (
            edited_chosen is None
            and self.web_search_enabled
            and confidence < self.web_search_threshold
        ):
            web_result = self._web_search(question)
            if web_result.get("found"):
                web_source = {
                    "source_url": web_result.get("source_url"),
                    "snippet": web_result.get("snippet"),
                    "answer": web_result.get("answer"),
                }

                # Step 3: Re-judge with web-sourced answer
                corrected = self._formulate_corrected(
                    question, web_result["answer"], web_source
                )
                edited_chosen = corrected
                edited_rejected = chosen if not chosen_correct else rejected

                # Re-evaluate confidence
                rejudge = self._judge_answer(question, corrected, web_result["answer"])
                confidence = max(confidence, rejudge.get("confidence", 0.7))
                explanation += f" Web search correction applied (source: {web_result.get('source_url', 'unknown')})."

        # Step 4: Determine action
        if edited_chosen is not None:
            action = "edit"
        elif confidence >= 0.7 and chosen_correct:
            action = "accept"
        elif confidence < 0.3:
            action = "reject"
        else:
            action = "accept"  # default to accept if no strong signal to reject

        return {
            "action": action,
            "confidence": confidence,
            "explanation": explanation,
            "edited_chosen": edited_chosen,
            "edited_rejected": edited_rejected,
            "web_source": web_source,
            "reasoning": reasoning,
            "chosen_verdict": chosen_verdict,
            "rejected_verdict": rejected_verdict,
            "cross_check_passed": cross_check.get("cross_check_passed", True),
        }

    def review_batch(self, entries, on_progress=None):
        """Review multiple training entries.

        Includes a small delay between judge LLM calls to avoid hitting
        API rate limits (OpenAI: 500 RPM, OpenRouter: varies).

        Args:
            entries: List of entry dicts
            on_progress: Optional callback(current, total, result)

        Returns:
            List of review result dicts
        """
        import time as _time

        results = []
        total = len(entries)

        for i, entry in enumerate(entries):
            result = self.review_entry(entry)
            results.append(result)

            if on_progress:
                on_progress(i + 1, total, result)

            # Rate limit: small delay between LLM calls to avoid API throttling.
            # 0.5s delay = max 120 calls/min, well under OpenAI's 500 RPM limit.
            if i < total - 1 and self.judge_adapter is not None:
                _time.sleep(0.5)

        return results

    def _judge_answer(self, question, model_answer, claimed_correct=None):
        """Use the judge LLM to evaluate if an answer is correct.

        Args:
            question: The question text
            model_answer: The answer to evaluate
            claimed_correct: The claimed correct answer (if known)

        Returns:
            dict: {correct: bool, confidence: float, explanation: str, reasoning: str}
        """
        if self.judge_adapter is None:
            # No judge available — return neutral
            return {
                "correct": True,
                "confidence": 0.5,
                "explanation": "No judge adapter available. Cannot verify correctness.",
                "reasoning": "No judge adapter available. Cannot verify correctness.",
            }

        prompt = self._build_judge_prompt(question, model_answer, claimed_correct)

        try:
            response = self.judge_adapter.ask(prompt, max_tokens=512)
            return self._parse_judge_response(response)
        except Exception as e:
            logger.warning(f"Judge LLM call failed: {e}")
            return {
                "correct": True,
                "confidence": 0.5,
                "explanation": f"Judge call failed: {e}",
                "reasoning": f"Judge call failed: {e}",
            }

    def _build_judge_prompt(self, question, model_answer, claimed_correct):
        """Build the prompt sent to the judge LLM."""
        lines = [
            "You are an expert reviewer evaluating a training data entry for correctness.",
            "Analyze whether the provided answer is correct.",
            "Explain your logic step-by-step in the reasoning field before giving the verdict fields.",
            "",
            f"Question: {question}",
            "",
            f"Answer to evaluate: {model_answer}",
            "",
        ]
        if claimed_correct:
            lines.append(f"Claimed correct answer: {claimed_correct}")
            lines.append("")
        lines.extend([
            "Evaluate whether the answer is factually correct.",
            "Respond in the following JSON format only:",
            '{"reasoning": "step-by-step logic", "correct": true/false, "confidence": 0.0-1.0, "explanation": "brief verdict summary"}',
        ])
        return "\n".join(lines)

    def _cross_verify(self, question, chosen, rejected, claimed_correct=None):
        """Judge chosen and rejected answers independently and compare verdicts."""
        chosen_verdict = self._judge_answer(question, chosen, claimed_correct)
        rejected_verdict = self._judge_answer(question, rejected, claimed_correct)

        chosen_wrong = chosen_verdict.get("correct") is False
        rejected_right = rejected_verdict.get("correct") is True
        should_edit = chosen_wrong and rejected_right

        chosen_reasoning = chosen_verdict.get("reasoning") or chosen_verdict.get("explanation", "")
        rejected_reasoning = rejected_verdict.get("reasoning") or rejected_verdict.get("explanation", "")
        reasoning = (
            f"Chosen answer reasoning: {chosen_reasoning}\n"
            f"Rejected answer reasoning: {rejected_reasoning}"
        ).strip()

        return {
            "chosen_verdict": chosen_verdict,
            "rejected_verdict": rejected_verdict,
            "cross_check_passed": not should_edit,
            "should_edit": should_edit,
            "reasoning": reasoning,
        }

    def _parse_judge_response(self, response):
        """Parse the judge LLM response into a structured verdict."""
        # Try to extract a JSON object from the response.
        # Use a brace-matching approach instead of regex to handle
        # nested braces inside string values (e.g., explanations
        # that mention {A} or {B}).
        try:
            # First, try parsing the entire response as JSON
            data = json.loads(response.strip())
            return self._verdict_from_json(data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try to find a JSON object by brace matching
        try:
            start = response.find("{")
            while start != -1:
                depth = 0
                for i in range(start, len(response)):
                    if response[i] == "{":
                        depth += 1
                    elif response[i] == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = response[start:i + 1]
                            try:
                                data = json.loads(candidate)
                                return self._verdict_from_json(data)
                            except (json.JSONDecodeError, ValueError, TypeError):
                                break  # try next start
                start = response.find("{", start + 1)
        except Exception:
            pass

        logger.warning("Failed to parse judge response as JSON, using text fallback.")

        # Fallback: infer from text
        response_lower = response.lower()
        if "correct" in response_lower and "incorrect" not in response_lower:
            text = response.strip()[:500]
            return {"correct": True, "confidence": 0.7, "explanation": text, "reasoning": text}
        elif "incorrect" in response_lower or "wrong" in response_lower:
            text = response.strip()[:500]
            return {"correct": False, "confidence": 0.3, "explanation": text, "reasoning": text}
        else:
            text = response.strip()[:500]
            return {"correct": False, "confidence": 0.5, "explanation": text, "reasoning": text}

    def _verdict_from_json(self, data):
        """Normalize judge JSON into the internal verdict shape."""
        explanation = str(data.get("explanation") or "No explanation provided.")
        reasoning = str(data.get("reasoning") or explanation)
        return {
            "correct": self._coerce_bool(data.get("correct", False)),
            "confidence": float(data.get("confidence", 0.5)),
            "explanation": explanation,
            "reasoning": reasoning,
        }

    @staticmethod
    def _coerce_bool(value):
        """Coerce common JSON/string boolean values without treating 'false' as true."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "correct"}:
                return True
            if normalized in {"false", "no", "incorrect", "wrong"}:
                return False
        return bool(value)

    def _web_search(self, query):
        """Search the web for the correct answer.

        Uses DuckDuckGo Lite (no API key needed) only to discover source URLs,
        then fetches result pages and extracts readable answer content from them.

        Args:
            query: Search query string

        Returns:
            dict: {found: bool, answer: str, source_url: str, snippet: str}
        """
        result = {
            "found": False,
            "answer": "",
            "source_url": "",
            "snippet": "",
            "candidates": [],
        }

        if requests is None:
            logger.warning("requests library not available. Web search disabled.")
            return result

        try:
            from html import unescape

            def extract_page_units(page_html):
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(page_html or "", "html.parser")
                    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form"]):
                        tag.decompose()

                    roots = soup.find_all(["main", "article"]) or [soup.body or soup]
                    units = []
                    for root in roots:
                        for node in root.find_all(["p", "li", "blockquote", "h1", "h2", "h3"]):
                            text = self._clean_text(node.get_text(" ", strip=True))
                            if len(text) >= 20:
                                units.append(text)
                    if units:
                        return units

                    body_text = self._clean_text((soup.body or soup).get_text(" ", strip=True))
                    return [body_text] if len(body_text) >= 20 else []
                except ImportError:
                    text = re.sub(
                        r"(?is)<(script|style|noscript|svg|header|footer|nav|aside|form)[^>]*>.*?</\1>",
                        " ",
                        page_html or "",
                    )
                    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
                    text = re.sub(
                        r"(?i)</(p|li|blockquote|h[1-6]|div|section|article|main)>",
                        "\n",
                        text,
                    )
                    text = re.sub(r"(?s)<[^>]+>", " ", text)
                    text = unescape(text)
                    return [
                        self._clean_text(unit)
                        for unit in re.split(r"\n+", text)
                        if len(self._clean_text(unit)) >= 20
                    ]

            def split_candidate_units(units):
                candidates = []
                for unit in units:
                    if len(unit) <= 500:
                        candidates.append(unit)
                        continue
                    sentences = [
                        self._clean_text(sentence)
                        for sentence in re.split(r"(?<=[.!?])\s+", unit)
                    ]
                    candidates.extend(sentence for sentence in sentences if len(sentence) >= 20)
                return candidates

            def fetch_page_candidate(url):
                try:
                    page_resp = requests.get(url, headers=headers, timeout=5)
                except requests.exceptions.Timeout:
                    logger.debug(f"Timed out fetching search result: {url}")
                    return None
                except Exception as e:
                    logger.debug(f"Failed to fetch search result {url}: {e}")
                    return None

                if page_resp.status_code != 200:
                    logger.debug(f"Search result returned status {page_resp.status_code}: {url}")
                    return None

                content_type = ""
                headers_obj = getattr(page_resp, "headers", None)
                if headers_obj:
                    raw_content_type = headers_obj.get("Content-Type", "")
                    if isinstance(raw_content_type, str):
                        content_type = raw_content_type.lower()
                if "pdf" in content_type:
                    return None

                units = split_candidate_units(extract_page_units(page_resp.text))
                if not units:
                    return None

                answer = max(
                    units,
                    key=lambda unit: (
                        self._score_answer_candidate(query, unit),
                        min(len(unit), 500),
                    ),
                )
                return answer[:500]

            # Use DuckDuckGo Lite (no API key needed)
            search_url = "https://lite.duckduckgo.com/lite/"
            params = {"q": query, "kl": "us-en"}
            headers = {
                "User-Agent": "MindForge/7.0 (automated review; +https://github.com/nousresearch)"
            }

            resp = requests.post(search_url, data=params, headers=headers, timeout=10)

            if resp.status_code != 200:
                logger.warning(f"Web search returned status {resp.status_code}")
                return result

            search_results = self._extract_search_results(resp.text, search_url)
            if not search_results:
                return result

            answer_candidates = []
            for candidate in search_results[:3]:
                answer = fetch_page_candidate(candidate["url"])
                if answer:
                    answer_candidates.append({
                        "url": candidate["url"],
                        "source_url": candidate["url"],
                        "answer": answer,
                        "snippet": answer[:500],
                        "confidence": self._score_answer_candidate(query, answer),
                    })

            if answer_candidates:
                answer_candidates.sort(key=lambda item: item["confidence"], reverse=True)
                best = answer_candidates[0]
                result["found"] = True
                result["source_url"] = best["url"]
                result["snippet"] = best["snippet"]
                result["answer"] = best["answer"]
                result["candidates"] = answer_candidates
                logger.info(f"Web search extracted answer content from: {best['url']}")
                return result

        except requests.exceptions.Timeout:
            logger.warning("Web search timed out.")
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

        return result

    def _extract_search_results(self, html, base_url):
        """Extract normalized, non-DuckDuckGo result URLs from search HTML."""
        links = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                links.append({
                    "url": urljoin(base_url, link["href"]),
                    "text": self._clean_text(link.get_text(" ", strip=True)),
                    "snippet": "",
                })
        except ImportError:
            parser = _LinkExtractor(base_url)
            parser.feed(html)
            links = parser.links

        results = []
        seen = set()
        for link in links:
            url = self._normalize_result_url(link["url"])
            if not self._is_usable_result_url(url) or url in seen:
                continue
            seen.add(url)
            results.append({
                "url": url,
                "text": link.get("text", ""),
                "snippet": link.get("snippet", ""),
            })
        return results

    def _fetch_result_answer(self, url, headers):
        """Fetch one result URL and extract readable answer text."""
        try:
            resp = requests.get(url, headers=headers, timeout=5)
        except requests.exceptions.Timeout:
            logger.debug(f"Timed out fetching search result: {url}")
            return ""
        except Exception as e:
            logger.debug(f"Failed to fetch search result {url}: {e}")
            return ""

        if resp.status_code != 200:
            logger.debug(f"Search result returned status {resp.status_code}: {url}")
            return ""

        content_type = ""
        headers_obj = getattr(resp, "headers", None)
        if headers_obj:
            raw_content_type = headers_obj.get("Content-Type", "")
            if isinstance(raw_content_type, str):
                content_type = raw_content_type.lower()
        if "pdf" in content_type:
            return ""

        return self._extract_answer_content(resp.text)

    def _score_answer_candidate(self, query, answer):
        """Score an extracted page answer against the review question."""
        query_terms = set(self._meaningful_terms(query))
        if not query_terms:
            return 0.5

        answer_terms = set(self._meaningful_terms(answer))
        overlap = len(query_terms & answer_terms) / len(query_terms)
        score = 0.2 + (0.7 * overlap)

        query_lower = query.lower()
        answer_lower = answer.lower()
        if "capital" in query_terms:
            location = self._extract_capital_query_location(query_lower)
            if location and f"capital of {location}" in answer_lower:
                score += 0.2
        if query_lower.strip(" ?.") in answer_lower:
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _meaningful_terms(text):
        """Return lower-cased terms with common question words removed."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "for", "from", "how", "in",
            "is", "of", "on", "or", "the", "to", "what", "when", "where",
            "which", "who", "why",
        }
        return [
            term for term in re.findall(r"[a-z0-9]+", text.lower())
            if term not in stopwords and len(term) > 1
        ]

    @staticmethod
    def _extract_capital_query_location(query):
        """Extract a simple location phrase from questions like 'capital of France'."""
        match = re.search(r"\bcapital\s+of\s+([a-z][a-z\s]+?)(?:\?|\.|$)", query)
        if not match:
            return ""
        return AutoReviewer._clean_text(match.group(1))

    def _extract_answer_content(self, html):
        """Extract readable page content for use as evidence-backed answer text."""
        fragments = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "form"]):
                tag.decompose()

            containers = soup.select("article, main, [role='main']")
            if not containers:
                containers = [soup]

            for container in containers:
                for node in container.find_all(["p", "li", "blockquote", "h1", "h2", "h3"]):
                    text = self._clean_text(node.get_text(" ", strip=True))
                    if text:
                        fragments.append(text)
        except ImportError:
            parser = _ReadableTextExtractor()
            parser.feed(html)
            fragments = parser.fragments

        return self._join_answer_fragments(fragments)

    @staticmethod
    def _normalize_result_url(url):
        """Decode DuckDuckGo redirect URLs to their real targets."""
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
        return url

    @staticmethod
    def _is_usable_result_url(url):
        """Return True for external HTTP(S) URLs that can be fetched as sources."""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if parsed.scheme not in {"http", "https"}:
            return False
        if not host or "duckduckgo.com" in host:
            return False
        return True

    @staticmethod
    def _clean_text(text):
        """Normalize whitespace in extracted text."""
        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def _join_answer_fragments(cls, fragments, max_chars=1200):
        """Join the best readable fragments into a concise answer body."""
        selected = []
        seen = set()
        total = 0
        for fragment in fragments:
            text = cls._clean_text(fragment)
            if len(text) < 20 or text in seen:
                continue
            seen.add(text)
            if total + len(text) + 1 > max_chars:
                remaining = max_chars - total
                if remaining > 80:
                    selected.append(text[:remaining].rsplit(" ", 1)[0])
                break
            selected.append(text)
            total += len(text) + 1
            if len(selected) >= 4:
                break
        return " ".join(selected)

    def _formulate_corrected(self, question, correct_answer, web_source=None):
        """Create a corrected answer text using the web-sourced answer.

        Args:
            question: The original question
            correct_answer: The correct answer text (from web search or judge)
            web_source: Optional dict with source_url, snippet

        Returns:
            str: Corrected answer text with source attribution
        """
        lines = [f"The answer is: {correct_answer}"]

        if web_source and web_source.get("source_url"):
            lines.append("")
            lines.append(f"Source: {web_source['source_url']}")

        return "\n".join(lines)

    def close(self):
        """Clean up adapter resources."""
        if self.judge_adapter:
            self.judge_adapter.close()
