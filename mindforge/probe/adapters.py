"""Model adapters for probing different model backends."""

import re
import os
import logging

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Base class for model adapters."""

    def __init__(self, model_name):
        """Initialize the adapter with a model name.

        Args:
            model_name: HuggingFace repo, OpenAI model ID, or exo model path.
        """
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
        """Initialize the MLX adapter (model loads lazily on first ask)."""
        super().__init__(model_name)
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        """Lazy-load the MLX model and tokenizer on first use."""
        if self.model is None:
            logger.info(f"Loading MLX model: {self.model_name}")
            from mlx_lm import load
            self.model, self.tokenizer = load(self.model_name)
            logger.info("Model loaded successfully.")

    def ask(self, question, max_tokens=512):
        """Generate a response from the MLX model for the given question."""
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
        """Release the MLX model and tokenizer from memory."""
        self.model = None
        self.tokenizer = None


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI API models."""

    def __init__(self, model_name):
        """Initialize the OpenAI adapter (client created lazily)."""
        super().__init__(model_name)
        self.client = None

    def _ensure_client(self):
        """Create the OpenAI client on first use."""
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()

    def ask(self, question, max_tokens=512):
        """Send a question to the OpenAI API and return the response."""
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
        """Initialize the OpenRouter adapter (client created lazily)."""
        super().__init__(model_name)
        self.client = None

    def _ensure_client(self):
        """Create the OpenAI client pointed at OpenRouter on first use."""
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )

    def ask(self, question, max_tokens=512):
        """Send a question to the OpenRouter API and return the response."""
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
        """Initialize the Exo adapter with default cluster endpoint."""
        super().__init__(model_name)
        self.client = None
        self.base_url = self.EXO_BASE_URL
        self.api_key = self.EXO_API_KEY

    def _ensure_client(self):
        """Create the OpenAI client pointed at the exo cluster on first use."""
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    def ask(self, question, max_tokens=512):
        """Send a question through the exo cluster and return the response."""
        self._ensure_client()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def close(self):
        """Release the exo client connection."""
        self.client = None


class OllamaAdapter(ModelAdapter):
    """Adapter for local Ollama models using Ollama's OpenAI-compatible API.

    Ollama is a local model runner that exposes an OpenAI-compatible
    API endpoint at http://localhost:11434/v1. This adapter routes
    requests through Ollama, allowing any GGUF model managed by
    `ollama pull` to be used for probing.
    """

    # Default Ollama API settings
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    OLLAMA_API_KEY = "ollama"

    def __init__(self, model_name):
        """Initialize the Ollama adapter (strips ollama/ prefix, client lazy)."""
        super().__init__(model_name)
        # Strip the ollama/ prefix — Ollama's API expects the bare model name
        if model_name.startswith("ollama/"):
            model_name = model_name[len("ollama/"):]
        self.model_name = model_name
        self.client = None
        self.base_url = self.OLLAMA_BASE_URL
        self.api_key = self.OLLAMA_API_KEY

    def _ensure_client(self):
        """Create the OpenAI client pointed at the local Ollama server on first use."""
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    def ask(self, question, max_tokens=512):
        """Ask the Ollama model a question and return the response string."""
        self._ensure_client()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def close(self):
        """Release the Ollama client connection."""
        self.client = None


def create_adapter(model_name):
    """Factory function to create the right adapter for a model.

    - MLX models (mlx-community/...) -> MLXAdapter
    - OpenAI models (gpt-...) -> OpenAIAdapter
    - OpenRouter models (openrouter/...) -> OpenRouterAdapter
    - Exo cluster models (exo/...) -> ExoAdapter
    - Ollama models (ollama/...) -> OllamaAdapter
    - Exo detected and running with peers -> ExoAdapter for local models

    When exo is running with peers, local MLX models are automatically
    routed through the ExoAdapter to leverage cluster memory.
    """
    # Check for explicit exo/ prefix
    if model_name.startswith("exo/"):
        return ExoAdapter(model_name)

    # Check for explicit ollama/ prefix
    if model_name.startswith("ollama/"):
        return OllamaAdapter(model_name)

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
