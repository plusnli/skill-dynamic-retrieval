"""Unified LiteLLM/vLLM client."""

import os
import base64
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

BACKEND = os.environ.get("LLM_BACKEND", "litellm").lower()
assert BACKEND in ("litellm", "vllm"), (
    f"LLM_BACKEND must be 'litellm' or 'vllm', got '{BACKEND}'"
)


def _get_vllm_client() -> OpenAI:
    """Create a vLLM client."""
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY", "EMPTY")
    if not base_url:
        raise ValueError(
            "LLM_BACKEND=vllm requires LLM_BASE_URL "
            "(e.g. http://localhost:8800/v1)"
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def _resolve_model(caller_model: str) -> str:
    """Resolve backend model name."""
    override = os.environ.get("LLM_MODEL_NAME")
    if override:
        return override
    if BACKEND == "vllm":
        model = caller_model
        for prefix in ("litellm/", "openai/", "neulab/"):
            model = model.replace(prefix, "")
        return model
    return caller_model.replace("litellm", "openai")


def llm_completion(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    n: int = 1,
    **kwargs,
):
    """Unified completion call. Returns OpenAI-style response object."""
    resolved_model = _resolve_model(model)

    if BACKEND == "litellm":
        import litellm
        return litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=os.environ.get("LITELLM_BASE_URL", "https://cmu.litellm.ai"),
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            n=n,
            **kwargs,
        )

    # vLLM backend.
    client = _get_vllm_client()
    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        n=n,
        **kwargs,
    )
    return response


# Autoeval clients.

class LM_Client:
    """Text-only eval client."""

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name

    def chat(self, messages, json_mode: bool = False) -> tuple:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = llm_completion(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            **kwargs,
        )
        return response.choices[0].message.content, response

    def one_step_chat(self, text, system_msg: str = None, json_mode=False) -> tuple:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode)


class GPT4V_Client:
    """Vision eval client."""

    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens

    @staticmethod
    def encode_image(path: str) -> str:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def one_step_chat(self, text, image, system_msg=None) -> tuple:
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"}},
            ],
        })
        response = llm_completion(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content, response
