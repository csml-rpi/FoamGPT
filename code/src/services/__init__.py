# Namespace package for service-layer wrappers

from __future__ import annotations

from typing import Optional

from utils import LLMService
from config import Config


# Single shared LLMService to avoid duplicate model instantiation (esp. local vLLM/HF).
# main.py should call set_global_llm_service(LLMService(config)) before execution.

_global_llm_service: Optional[LLMService] = None


def set_global_llm_service(svc: LLMService) -> None:
    global _global_llm_service
    _global_llm_service = svc


def get_global_llm_service() -> LLMService:
    global _global_llm_service
    if _global_llm_service is None:
        _global_llm_service = LLMService(Config())
    return _global_llm_service


class _LazyLLMProxy:
    def invoke(self, *args, **kwargs):
        return get_global_llm_service().invoke(*args, **kwargs)

    def get_statistics(self):
        return get_global_llm_service().get_statistics()

    def print_statistics(self):
        return get_global_llm_service().print_statistics()


# Keep name for existing imports.
global_llm_service = _LazyLLMProxy()
