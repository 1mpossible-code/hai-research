"""OpenAI backend"""

import os
import time
from typing import Any

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class OpenAIBackend:
    """OpenAI API backend."""
    
    def __init__(self):
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable not set. "
                        "Set it or remove OpenAI models from config."
                    )
                api_base = os.getenv("OPENAI_API_BASE")  # Optional for custom endpoints
                if api_base:
                    self._client = OpenAI(api_key=api_key, base_url=api_base)
                else:
                    self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install -e .[openai]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        return self._client
    
    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Call OpenAI API with retry logic."""
        client = self._get_client()
        max_retries = 5
        backoff_base = 1.0
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_spec.name,
                    messages=[
                        {"role": "user", "content": req.prompt}
                    ],
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                
                raw_output = response.choices[0].message.content or ""
                
                # Extract usage
                usage = None
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    }
                
                return JudgeResponse(
                    raw_output=raw_output,
                    usage=usage,
                    status="ok",
                )
            
            except Exception as e:
                error_str = str(e).lower()
                if ("rate limit" in error_str or "429" in error_str) and attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt) + (time.time() % 1)
                    time.sleep(wait_time)
                    continue
                
                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt) + (time.time() % 1)
                    time.sleep(wait_time)
                    continue
                
                return JudgeResponse(
                    raw_output="",
                    usage=None,
                    status="error",
                    error=str(e),
                )
        
        return JudgeResponse(
            raw_output="",
            usage=None,
            status="error",
            error="Max retries exceeded",
        )

