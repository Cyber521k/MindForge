"""Model adapters for probing different model backends."""

import re
import os
import logging

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Base class for model adapters."""

    def __init__(self, model_name):
        self.model_name = model_name

    def ask(self, question, max_tokens=512):
        """Ask the model a question and return the response string."""
        raise NotImplementedError

    def close(self):
        """Clean up resources."""
        pass


class MLXAdapter(ModelAdapter):
    """Adapter for local MLX models using mlx_lm."""

    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        if self.model is None:
            logger.info(f"Loading MLX model: {self.model_name}")
            from mlx_lm import load
            self.model, self.tokenizer = load(self.model_name)
            logger.info("Model loaded successfully.")

    def ask(self, question, max_tokens=512):
        self._ensure_loaded()
        from mlx_lm import generate

        # Use chat template if available
        try:
            messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = question

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response

    def close(self):
        self.model = None
        self.tokenizer = None


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI API models."""

    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = None

    def _ensure_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()

    def ask(self, question, max_tokens=512):
        self._ensure_client()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class OpenRouterAdapter(ModelAdapter):
    """Adapter for OpenRouter API models."""

    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = None

    def _ensure_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )

    def ask(self, question, max_tokens=512):
        self._ensure_client()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class ExoAdapter(ModelAdapter):
    """Adapter for Exo cluster models using exo's OpenAI-compatible API.

    Exo is a distributed inference framework that exposes an OpenAI-compatible
    API endpoint. This adapter routes requests through the exo cluster,
    allowing models larger than a single machine's memory to be served
    across multiple Apple Silicon devices.
    """

    # Default exo API settings
    EXO_BASE_URL = "http://localhost:52415/v1"
    EXO_API_KEY = "exo"

    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = None
        self.base_url = self.EXO_BASE_URL
        self.api_key = self.EXO_API_KEY

    def _ensure_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    def ask(self, question, max_tokens=512):
        self._ensure_client()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def close(self):
        self.client = None


def create_adapter(model_name):
    """Factory function to create the right adapter for a model.

    - MLX models (mlx-community/...) -> MLXAdapter
    - OpenAI models (gpt-...) -> OpenAIAdapter
    - OpenRouter models (openrouter/...) -> OpenRouterAdapter
    - Exo cluster models (exo/...) -> ExoAdapter
    - Exo detected and running with peers -> ExoAdapter for local models

    When exo is running with peers, local MLX models are automatically
    routed through the ExoAdapter to leverage cluster memory.
    """
    # Check for explicit exo/ prefix
    if model_name.startswith("exo/"):
        return ExoAdapter(model_name)

    # Check if exo is running with peers - if so, route local models through exo
    try:
        from mindforge.hardware.exo_detector import detect_exo
        exo_info = detect_exo()
        if exo_info.get("running") and exo_info.get("peer_count", 0) > 0:
            # Route MLX models through exo cluster
            if model_name.startswith("mlx-community/") or model_name.startswith("mlx-"):
                return ExoAdapter(model_name)
    except Exception:
        # If exo detection fails, fall back to normal adapter selection
        pass

    if model_name.startswith("mlx-community/") or model_name.startswith("mlx-"):
        return MLXAdapter(model_name)
    elif model_name.startswith("openrouter/"):
        return OpenRouterAdapter(model_name)
    elif model_name.startswith("gpt-") or model_name.startswith("o1-") or model_name.startswith("o3-"):
        return OpenAIAdapter(model_name)
    else:
        # Default to MLX for unknown models on Apple Silicon
        return MLXAdapter(model_name)


def extract_answer_letter(response):
    """Extract A, B, C, or D from a model response.

    Tries multiple patterns to find the answer letter.
    """
    response = response.strip()

    # Pattern 1: "The answer is B" or "answer is B"
    match = re.search(r"(?:answer\s+is\s*)([ABCD])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: "Answer: B"
    match = re.search(r"(?:answer\s*:\s*)([ABCD])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Just the letter at the start or standalone
    match = re.search(r"\b([ABCD])\b", response)
    if match:
        return match.group(1).upper()

    # Pattern 4: "(B)" format
    match = re.search(r"\(([ABCD])\)", response)
    if match:
        return match.group(1).upper()

    # Pattern 5: "B)" at start of line
    match = re.search(r"^([ABCD])\)", response, re.MULTILINE)
    if match:
        return match.group(1).upper()

    return None
