"""HuggingFace local Llama backend"""

import os
import torch
from typing import Any

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


# Global model cache
_model_cache: dict[str, tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)


class HuggingFaceBackend:
    """HuggingFace transformers backend for local models."""
    
    def __init__(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package not installed. Install with: pip install -e .[hf]"
            )
    
    def _get_model(self, model_spec: ModelSpec):
        """Get or load model and tokenizer (cached)."""
        model_name = model_spec.name
        
        if model_name in _model_cache:
            return _model_cache[model_name]
        
        device = model_spec.params.get("device", "cpu")
        dtype_str = model_spec.params.get("dtype", "float16")
        
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.float16)
        
        tokenizer = self.AutoTokenizer.from_pretrained(model_name)
        model = self.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        _model_cache[model_name] = (model, tokenizer)
        return model, tokenizer
    
    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Generate judgment using local model."""
        import torch
        
        try:
            model, tokenizer = self._get_model(model_spec)
            
            # Format prompt (use chat template if available)
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": req.prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback formatting
                formatted_prompt = req.prompt
            
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with deterministic-ish seed
            max_new_tokens = model_spec.params.get("max_new_tokens", req.max_tokens)
            
            # Set seed for reproducibility (based on prompt hash for determinism)
            import hashlib
            seed = int(hashlib.md5(req.text.encode()).hexdigest()[:8], 16) % (2**31)
            torch.manual_seed(seed)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=req.temperature if req.temperature > 0 else None,
                    do_sample=req.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode only new tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return JudgeResponse(
                raw_output=raw_output,
                usage=None,  # Local models don't provide usage
                status="ok",
            )
        
        except Exception as e:
            return JudgeResponse(
                raw_output="",
                usage=None,
                status="error",
                error=str(e),
            )

