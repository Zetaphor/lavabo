from __future__ import annotations

# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

from huggingface_hub import hf_hub_download  # type: ignore[reportMissingTypeStubs]
from pydantic import BaseModel

try:
    # Imported lazily to allow pyproject parsing without native deps present
    from llama_cpp import Llama  # type: ignore[reportMissingTypeStubs]
except Exception:  # pragma: no cover - import may fail before env is ready
    Llama = None  # type: ignore


StructuredT = TypeVar("StructuredT", bound=BaseModel)


@dataclass(frozen=True)
class InferenceResult:
    text: str
    raw: Dict[str, Any]


class GGUFModelLoader:
    """Load and run GGUF models via llama-cpp-python.

    Supports loading from local file paths or Hugging Face repos. Provides
    free-form generation and optional structured outputs via Pydantic schemas.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not available. Install extras before use."
            )

        model_path = str(model_path)
        init_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "verbose": verbose,
        }
        if n_threads is not None:
            init_kwargs["n_threads"] = n_threads

        try:
            self._llm = Llama(**init_kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            # If GPU offload is requested but unavailable, fall back to CPU
            if n_gpu_layers and n_gpu_layers != 0:
                warnings.warn(
                    f"GPU offload initialization failed ({exc!s}); falling back to CPU.",
                    RuntimeWarning,
                )
                init_kwargs["n_gpu_layers"] = 0
                self._llm = Llama(**init_kwargs)
            else:
                raise

    # ---------- Loading helpers ----------
    @classmethod
    def from_path(
        cls,
        model_path: Union[str, Path],
        **kwargs: Any,
    ) -> "GGUFModelLoader":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return cls(model_path=path, **kwargs)

    @classmethod
    def from_hf(
        cls,
        *,
        repo_id: str,
        filename: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "GGUFModelLoader":
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return cls(model_path=local_path, **kwargs)

    # ---------- Inference ----------
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[Union[str, list[str]]] = None,
        echo: bool = False,
        schema: Optional[Type[StructuredT]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        format_json: bool = False,
        add_assistant_prefix: bool = False,
        **kwargs: Any,
    ) -> Union[str, StructuredT, InferenceResult]:
        """Generate text or structured output.

        If `schema` or `json_schema` are provided, the model is nudged to
        produce a JSON object matching the given schema. The output is parsed
        and validated into the provided Pydantic model when `schema` is given.
        When structured output is requested, returns the parsed model instance
        (or the raw JSON dict if `schema` is None but `json_schema` provided).

        Otherwise, returns the generated string. If callers need both the text
        and the raw llama-cpp response, set `return_raw=True` via kwargs to
        obtain an `InferenceResult`.
        """

        enforce_json = schema is not None or json_schema is not None or format_json

        formatted_prompt = prompt
        if enforce_json:
            import json as _json

            system_hint = "You are a parser. Return ONLY valid minified JSON matching the given schema."
            schema_dict: Dict[str, Any] = {}
            if schema is not None:
                schema_dict = cast(Dict[str, Any], schema.model_json_schema())
            elif json_schema is not None:
                schema_dict = json_schema

            schema_json_str = _json.dumps(schema_dict, separators=(",", ":"))
            formatted_prompt = (
                f"{system_hint}\nSchema: {schema_json_str}\nInput: {prompt}\nOutput:"
            )

        if add_assistant_prefix:
            formatted_prompt = f"{formatted_prompt}\nAssistant:"

        response = cast(
            Dict[str, Any],
            self._llm(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=echo,
                stream=False,
                **kwargs,
            ),
        )

        text = response.get("choices", [{}])[0].get("text", "")

        if not enforce_json:
            if kwargs.get("return_raw"):
                return InferenceResult(text=text, raw=response)
            return text

        # Try to extract JSON from the response text
        import json

        parsed: Any
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to find the first JSON object in the text
            import re

            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise ValueError(
                    "Model did not return JSON; unable to parse structured output"
                )
            parsed = json.loads(match.group(0))

        if schema is not None:
            return schema.model_validate(parsed)

        if kwargs.get("return_raw"):
            return InferenceResult(text=text, raw=response)
        return parsed
