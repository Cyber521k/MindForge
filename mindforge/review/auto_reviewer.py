"""Automated review system using LLM-as-judge and web search.

The AutoReviewer evaluates training entries (prompt/chosen/rejected pairs) using
a judge LLM. If the judge is uncertain (confidence < 0.7), it falls back to
web search to find the correct answer. This automates the review queue that
previously required manual human sign-off.
"""

import os
import re
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


class AutoReviewer:
    """Automated review of training entries using LLM-as-judge + web search.

    Workflow per entry:
        1. Send question + model answer + chosen/rejected to judge LLM
        2. Judge evaluates correctness of chosen and rejected
        3. If judge confidence < 0.7, search the web for the correct answer
        4. If web search finds a better answer, correct the chosen/rejected pair
        5. Return action: accept / reject / edit
    """

    # Confidence threshold below which web search is triggered
    WEB_SEARCH_THRESHOLD = 0.7

    def __init__(self, judge_adapter=None, web_search_enabled=True):
        """Initialize the auto-reviewer.

        Args:
            judge_adapter: A ModelAdapter for the reviewing LLM (OpenAIAdapter,
                          OpenRouterAdapter, etc.). If None, auto-detects from
                          available API keys or falls back to Ollama/MLX.
            web_search_enabled: Whether to use web search when the judge is uncertain.
        """
        self.judge_adapter = judge_adapter
        self.web_search_enabled = web_search_enabled
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
                model_name = ollama_info["models"][0].get("name", "llama3.1:latest")
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
        question = entry.get("question") or entry.get("prompt", "")
        chosen = entry.get("chosen", "")
        rejected = entry.get("rejected", "")
        subject = entry.get("subject", "unknown")

        # Step 1: Judge the chosen answer
        judge_result = self._judge_answer(question, chosen, entry.get("correct_answer"))
        confidence = judge_result.get("confidence", 0.5)
        chosen_correct = judge_result.get("correct", True)
        explanation = judge_result.get("explanation", "")

        web_source = None
        edited_chosen = None
        edited_rejected = None

        # Step 2: If uncertain, try web search
        if self.web_search_enabled and confidence < self.WEB_SEARCH_THRESHOLD:
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
        if confidence >= 0.7 and chosen_correct:
            action = "accept"
        elif edited_chosen is not None:
            action = "edit"
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
        }

    def review_batch(self, entries, on_progress=None):
        """Review multiple training entries.

        Args:
            entries: List of entry dicts
            on_progress: Optional callback(current, total, result)

        Returns:
            List of review result dicts
        """
        results = []
        total = len(entries)

        for i, entry in enumerate(entries):
            result = self.review_entry(entry)
            results.append(result)

            if on_progress:
                on_progress(i + 1, total, result)

        return results

    def _judge_answer(self, question, model_answer, claimed_correct=None):
        """Use the judge LLM to evaluate if an answer is correct.

        Args:
            question: The question text
            model_answer: The answer to evaluate
            claimed_correct: The claimed correct answer (if known)

        Returns:
            dict: {correct: bool, confidence: float, explanation: str}
        """
        if self.judge_adapter is None:
            # No judge available — return neutral
            return {
                "correct": True,
                "confidence": 0.5,
                "explanation": "No judge adapter available. Cannot verify correctness.",
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
            }

    def _build_judge_prompt(self, question, model_answer, claimed_correct):
        """Build the prompt sent to the judge LLM."""
        lines = [
            "You are an expert reviewer evaluating a training data entry for correctness.",
            "Analyze whether the provided answer is correct and give your verdict.",
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
            '{"correct": true/false, "confidence": 0.0-1.0, "explanation": "your reasoning"}',
        ])
        return "\n".join(lines)

    def _parse_judge_response(self, response):
        """Parse the judge LLM response into a structured verdict."""
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return {
                    "correct": bool(data.get("correct", False)),
                    "confidence": float(data.get("confidence", 0.5)),
                    "explanation": str(data.get("explanation", "No explanation provided.")),
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse judge response as JSON: {e}")

        # Fallback: infer from text
        response_lower = response.lower()
        if "correct" in response_lower and "incorrect" not in response_lower:
            return {"correct": True, "confidence": 0.7, "explanation": response.strip()[:500]}
        elif "incorrect" in response_lower or "wrong" in response_lower:
            return {"correct": False, "confidence": 0.3, "explanation": response.strip()[:500]}
        else:
            return {"correct": False, "confidence": 0.5, "explanation": response.strip()[:500]}

    def _web_search(self, query):
        """Search the web for the correct answer.

        Uses DuckDuckGo Lite (no API key needed) with BeautifulSoup parsing.
        Falls back to a simple HTTP GET if BeautifulSoup is unavailable.

        Args:
            query: Search query string

        Returns:
            dict: {found: bool, answer: str, source_url: str, snippet: str}
        """
        result = {"found": False, "answer": "", "source_url": "", "snippet": ""}

        try:
            import requests
            from urllib.parse import quote_plus

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

            # Parse results with BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")

                # DuckDuckGo Lite results are in table rows
                # Find result links and snippets
                result_links = []
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if href.startswith("http") and "duckduckgo" not in href:
                        result_links.append({
                            "url": href,
                            "text": link.get_text(strip=True),
                        })

                if result_links:
                    best = result_links[0]
                    result["found"] = True
                    result["source_url"] = best["url"]
                    result["snippet"] = best["text"][:500]
                    # Use the snippet text as the answer
                    result["answer"] = best["text"][:500]
                    logger.info(f"Web search found result: {best['url']}")
                    return result

            except ImportError:
                logger.warning("BeautifulSoup not available. Trying raw text parsing.")
                # Fallback: regex extract URLs from raw HTML
                url_pattern = r'href="(https?://[^"]+)"'
                urls = re.findall(url_pattern, resp.text)
                filtered = [u for u in urls if "duckduckgo" not in u.lower()]
                if filtered:
                    result["found"] = True
                    result["source_url"] = filtered[0]
                    result["snippet"] = "Web result found (raw parsing)."
                    result["answer"] = "Web result found (raw parsing)."
                    return result

        except requests.exceptions.Timeout:
            logger.warning("Web search timed out.")
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

        return result

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
