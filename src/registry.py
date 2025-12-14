"""Backend registry for pluggable backends"""

from typing import Type

from src.judging.base import JudgeBackend
from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


def get_backend(backend_name: str) -> JudgeBackend:
    """Get backend instance by name.
    
    Args:
        backend_name: One of 'anthropic', 'openai', 'hf', 'mock'
    
    Returns:
        JudgeBackend instance
    """
    if backend_name == "mock":
        from src.judging.backends.mock_backend import MockBackend
        return MockBackend()
    
    elif backend_name == "anthropic":
        from src.judging.backends.anthropic_backend import AnthropicBackend
        return AnthropicBackend()
    
    elif backend_name == "openai":
        from src.judging.backends.openai_backend import OpenAIBackend
        return OpenAIBackend()
    
    elif backend_name == "hf":
        from src.judging.backends.hf_llama_backend import HuggingFaceBackend
        return HuggingFaceBackend()
    
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

