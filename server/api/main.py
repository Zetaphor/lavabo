# pyright: reportMissingTypeStubs=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import os
from dataclasses import dataclass
import json
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

import gc
import threading
import importlib
import numpy as np

# Import FastAPI dynamically to avoid linter issues when type stubs are missing
try:  # pragma: no cover
    _fastapi = importlib.import_module("fastapi")  # type: ignore
    FastAPI = _fastapi.FastAPI  # type: ignore
    HTTPException = _fastapi.HTTPException  # type: ignore
    try:
        StaticFiles = importlib.import_module("fastapi.staticfiles").StaticFiles  # type: ignore
    except Exception:  # pragma: no cover
        StaticFiles = None  # type: ignore
except Exception:  # pragma: no cover
    _fastapi = None  # type: ignore

    class _HTTPException(Exception):
        pass

    FastAPI = object  # type: ignore
    HTTPException = _HTTPException  # type: ignore
    StaticFiles = None  # type: ignore

from pydantic import BaseModel, Field

# pyright: reportInvalidTypeForm=false
from llama_cpp import Llama  # type: ignore


# ---------- Model configuration ----------


@dataclass(frozen=True)
class ModelConfig:
    name: str
    file: str
    chat_format: Optional[str]
    context_length: int


# ---------- App and lifecycle ----------

tags_metadata = [
    {
        "name": "health",
        "description": "Service health and readiness checks.",
    },
    {
        "name": "model",
        "description": "Model lifecycle management: load, unload, and status.",
    },
    {
        "name": "chat",
        "description": "Chat completions using the currently loaded model.",
    },
    {
        "name": "metrics",
        "description": "Runtime metrics such as GPU VRAM usage.",
    },
    {
        "name": "embeddings",
        "description": "Embedding model lifecycle and vector operations.",
    },
]

app = FastAPI(
    title="AI Toolkit - llama.cpp API",
    version="0.2.0",
    description=(
        "A minimal FastAPI wrapper around llama.cpp providing endpoints to load GGUF models, "
        "perform chat completions (optionally using a JSON response schema), and inspect VRAM usage."
    ),
    openapi_tags=tags_metadata,
    contact={
        "name": "AI Toolkit",
        "url": "https://github.com/zetaphor/ai-toolkit",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Mount static serving for generated TTS audio if FastAPI StaticFiles is available
try:
    from fastapi.staticfiles import StaticFiles as _StaticFiles

    _tts_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "tts", "output"
    )
    try:
        os.makedirs(_tts_output_dir, exist_ok=True)
    except Exception:
        pass
    app.mount("/audio", _StaticFiles(directory=_tts_output_dir), name="audio")
except Exception:
    pass


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ResponseFormat(BaseModel):
    type: Literal["json_object"] = Field(
        ...,
        description="Requested structured response mode. Currently only 'json_object' is supported.",
    )
    schema: Dict[str, Any] = Field(
        ..., description="JSON Schema for the object the model should return."
    )

    # Example shown in the OpenAPI schema
    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "sentiment": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": ["title", "sentiment", "score"],
                },
            }
        }
    }


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ..., description="Ordered chat messages in the conversation."
    )
    max_tokens: int = Field(
        default=256, description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature; higher values produce more random outputs.",
    )
    top_p: float = Field(default=0.95, description="Top-p nucleus sampling cutoff.")
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=(
            "Optional structured output configuration. When provided with type 'json_object', "
            "the server will attempt to parse the model output into a JSON object conforming to the schema."
        ),
    )

    # Example shown in the OpenAPI schema
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {
                            "role": "user",
                            "content": "Write a haiku about GPUs.",
                        },
                    ],
                    "max_tokens": 128,
                    "temperature": 0.7,
                },
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return only valid minified JSON.",
                        },
                        {
                            "role": "user",
                            "content": "Provide a brief summary with a title and sentiment score.",
                        },
                    ],
                    "response_format": {
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "sentiment": {"type": "string"},
                                "score": {"type": "number"},
                            },
                            "required": ["title", "sentiment", "score"],
                        },
                    },
                    "temperature": 0.2,
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    content: str
    raw: Optional[Dict[str, Any]] = None
    # If response_format.type == "json_object" and parsing succeeds
    parsed: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": '{"title":"GPU Haiku","sentiment":"positive","score":0.92}',
                "parsed": {
                    "title": "GPU Haiku",
                    "sentiment": "positive",
                    "score": 0.92,
                },
                "raw": {"choices": [{"message": {"content": "..."}}]},
            }
        }
    }


_llm: Optional[Llama] = None
_llm_lock = threading.Lock()
_loaded_cfg: Optional[ModelConfig] = None

# Embedding model globals
_emb_model = None  # type: ignore[var-annotated]
_emb_model_name: Optional[str] = None
_emb_lock = threading.Lock()


class LoadModelRequest(BaseModel):
    # One of: explicit file path OR a Hugging Face repo + file
    file: Optional[str] = Field(
        default=None, description="Absolute path to a GGUF file inside the container"
    )
    hf_repo: Optional[str] = Field(
        default=None,
        description="Hugging Face repo id, e.g. 'microsoft/Phi-3-mini-4k-instruct-gguf'",
    )
    hf_file: Optional[str] = Field(
        default=None,
        description="File name within the repo to download, e.g. 'Phi-3-mini-4k-instruct-q4.gguf'",
    )
    chat_format: Optional[str] = None
    n_ctx: Optional[int] = Field(
        default=None, description="Context length; default 4096 if not provided"
    )
    n_gpu_layers: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "file": "/models/microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
                    "n_ctx": 4096,
                    "n_gpu_layers": -1,
                    "chat_format": "chatml",
                },
                {
                    "hf_repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
                    "hf_file": "Phi-3-mini-4k-instruct-q4.gguf",
                    "n_ctx": 4096,
                    "chat_format": "chatml",
                },
            ]
        }
    }


class LoadModelResponse(BaseModel):
    status: str
    loaded: bool
    config: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "loaded",
                "loaded": True,
                "config": {
                    "name": "Phi-3-mini-4k-instruct-q4.gguf",
                    "file": "/models/microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
                    "chat_format": "chatml",
                    "n_ctx": 4096,
                },
            }
        }
    }


def _unload_llm_locked() -> None:
    global _llm, _loaded_cfg
    if _llm is not None:
        # Drop reference and force GC to release GPU memory
        _llm = None
        _loaded_cfg = None
        gc.collect()


@app.post(
    "/load_gguf",
    response_model=LoadModelResponse,
    summary="Load a GGUF model",
    description=(
        "Load a GGUF model either from an absolute file path inside the container or by "
        "downloading a file from Hugging Face Hub. If a model is already loaded, it will be unloaded first."
    ),
    tags=["model"],
    responses={
        400: {
            "description": "Bad request (missing parameters or file not found).",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Provide either 'file' or both 'hf_repo' and 'hf_file'"
                    }
                }
            },
        },
        500: {
            "description": "Server error (e.g., missing optional dependency).",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "huggingface_hub is required to download from HF: <reason>"
                    }
                }
            },
        },
    },
)
def load_model(req: LoadModelRequest) -> LoadModelResponse:
    global _llm, _loaded_cfg
    with _llm_lock:
        # Resolve config from either explicit file path or Hugging Face repo + file
        cfg: Optional[ModelConfig] = None
        target_chat_format = req.chat_format or "chatml"
        target_ctx = req.n_ctx or 4096

        if req.file:
            resolved_file = req.file
        elif req.hf_repo and req.hf_file:
            # Lazy import to keep optional dependency boundary
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise HTTPException(
                    status_code=500,
                    detail=f"huggingface_hub is required to download from HF: {exc}",
                )

            models_dir = os.environ.get("MODELS_DIR", "/models")
            # Ensure local dir exists
            try:
                os.makedirs(models_dir, exist_ok=True)
            except Exception:
                pass
            try:
                resolved_file = hf_hub_download(
                    repo_id=req.hf_repo,
                    filename=req.hf_file,
                    local_dir=models_dir,
                    local_dir_use_symlinks=False,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Failed to download from HF: {exc}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'file' or both 'hf_repo' and 'hf_file'",
            )

        if not os.path.exists(resolved_file):
            raise HTTPException(
                status_code=400, detail=f"Model file not found: {resolved_file}"
            )

        cfg = ModelConfig(
            name=os.path.basename(resolved_file),
            file=resolved_file,
            chat_format=target_chat_format,
            context_length=target_ctx,
        )

        # If already loaded, unload first
        _unload_llm_locked()

        # Initialize
        _llm = Llama(
            model_path=cfg.file,
            n_gpu_layers=(
                req.n_gpu_layers
                if req.n_gpu_layers is not None
                else int(os.environ.get("N_GPU_LAYERS", "-1"))
            ),
            n_ctx=cfg.context_length,
            chat_format=cfg.chat_format,
        )
        _loaded_cfg = cfg

        return LoadModelResponse(
            status="loaded",
            loaded=True,
            config={
                "name": cfg.name,
                "file": cfg.file,
                "chat_format": cfg.chat_format,
                "n_ctx": cfg.context_length,
            },
        )


@app.get(
    "/healthz",
    summary="Health check",
    description="Liveness probe endpoint to verify the service is running.",
    tags=["health"],
)
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat completion",
    description=(
        "Generate a chat completion using the currently loaded model. Optionally specify a structured "
        "response schema by setting response_format.type='json_object'."
    ),
    tags=["chat"],
    responses={
        503: {
            "description": "No model is currently loaded.",
            "content": {
                "application/json": {"example": {"detail": "Model not loaded yet"}}
            },
        },
        500: {
            "description": "Unexpected server error while generating completion.",
        },
    },
)
def chat(req: ChatRequest) -> ChatResponse:
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # llama-cpp-python types are not available; suppress type check noise here
        messages_payload: List[Dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in req.messages
        ]
        result: Dict[str, Any] = _llm.create_chat_completion(  # type: ignore[assignment]
            messages=messages_payload,  # type: ignore[arg-type]
            response_format=(
                req.response_format.model_dump() if req.response_format else None
            ),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=False,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))

    message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed: Optional[Dict[str, Any]] = None
    if req.response_format and req.response_format.type == "json_object":
        try:
            candidate = str(message or "").strip()
            parsed = json.loads(candidate) if candidate else None
            if parsed is not None and not isinstance(parsed, dict):
                # Enforce object type per json_object contract
                parsed = None
        except Exception:
            parsed = None

    return ChatResponse(content=str(message or ""), raw=result, parsed=parsed)


class UnloadResponse(BaseModel):
    status: str
    unloaded: bool

    model_config = {
        "json_schema_extra": {"example": {"status": "unloaded", "unloaded": True}}
    }


@app.post(
    "/unload_gguf",
    response_model=UnloadResponse,
    summary="Unload the current model",
    description="Unload the currently loaded GGUF model and free resources.",
    tags=["model"],
)
def unload_model() -> UnloadResponse:
    with _llm_lock:
        was_loaded = _llm is not None
        _unload_llm_locked()
        return UnloadResponse(status="unloaded", unloaded=was_loaded)


class StatusResponse(BaseModel):
    loaded: bool
    config: Optional[Dict[str, Any]]

    model_config = {
        "json_schema_extra": {
            "example": {
                "loaded": True,
                "config": {
                    "name": "Phi-3-mini-4k-instruct-q4.gguf",
                    "file": "/models/.../Phi-3-mini-4k-instruct-q4.gguf",
                    "chat_format": "chatml",
                    "n_ctx": 4096,
                },
            }
        }
    }


@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Model status",
    description="Return whether a model is loaded and its configuration if available.",
    tags=["model"],
)
def status() -> StatusResponse:
    return StatusResponse(
        loaded=_llm is not None,
        config=(
            {
                "name": _loaded_cfg.name if _loaded_cfg else None,
                "file": _loaded_cfg.file if _loaded_cfg else None,
                "chat_format": _loaded_cfg.chat_format if _loaded_cfg else None,
                "n_ctx": _loaded_cfg.context_length if _loaded_cfg else None,
            }
            if _loaded_cfg
            else None
        ),
    )


class VRAMDeviceUsage(BaseModel):
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    used_by_process_bytes: int


class VRAMUsageResponse(BaseModel):
    gpu_available: bool
    devices: List[VRAMDeviceUsage] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "gpu_available": True,
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA RTX 4090",
                        "total_bytes": 24576000000,
                        "free_bytes": 20000000000,
                        "used_by_process_bytes": 1024000000,
                    }
                ],
            }
        }
    }


@app.get(
    "/vram",
    response_model=VRAMUsageResponse,
    summary="GPU VRAM usage",
    description=(
        "Report per-device total and free VRAM, plus bytes used by this API process. "
    ),
    tags=["metrics"],
)
def vram_usage() -> VRAMUsageResponse:
    try:
        import pynvml  # type: ignore
    except Exception:
        return VRAMUsageResponse(gpu_available=False, devices=[])

    try:
        pynvml.nvmlInit()
    except Exception:
        return VRAMUsageResponse(gpu_available=False, devices=[])

    pid = os.getpid()
    device_count = pynvml.nvmlDeviceGetCount()
    devices: List[VRAMDeviceUsage] = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_by_proc = 0
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in procs:
                if getattr(proc, "pid", None) == pid:
                    used_by_proc += int(getattr(proc, "usedGpuMemory", 0) or 0)
        except Exception:
            used_by_proc = 0
        name = (
            pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            if hasattr(pynvml.nvmlDeviceGetName(handle), "decode")
            else str(pynvml.nvmlDeviceGetName(handle))
        )
        devices.append(
            VRAMDeviceUsage(
                index=i,
                name=name,
                total_bytes=int(mem_info.total),
                free_bytes=int(mem_info.free),
                used_by_process_bytes=used_by_proc,
            )
        )
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return VRAMUsageResponse(gpu_available=True, devices=devices)


# ------------------ TTS (Kokoro) API ------------------


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0


class TTSResponse(BaseModel):
    url: str
    path: str
    voice: str
    time_seconds: float


class TTSVoicesResponse(BaseModel):
    voices: List[str]


def _require_kokoro():
    try:
        from server.tts.model_setup import (
            generate_audio,
            AVAILABLE_VOICES,
            DEFAULT_VOICE,
        )  # type: ignore

        return generate_audio, AVAILABLE_VOICES, DEFAULT_VOICE
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Kokoro TTS not available: {exc}")


@app.get(
    "/tts/voices",
    response_model=TTSVoicesResponse,
    summary="List available Kokoro voices",
    description="Return the list of available Kokoro TTS voices.",
    tags=["tts"],
)
def tts_voices() -> TTSVoicesResponse:
    _, AVAILABLE_VOICES, _ = _require_kokoro()
    return TTSVoicesResponse(voices=list(AVAILABLE_VOICES))


@app.post(
    "/tts/synthesize",
    response_model=TTSResponse,
    summary="Synthesize speech with Kokoro",
    description=(
        "Generate a WAV file from input text using Kokoro TTS. Returns a URL to download the audio."
    ),
    tags=["tts"],
)
def tts_synthesize(req: TTSRequest) -> TTSResponse:
    generate_audio, AVAILABLE_VOICES, DEFAULT_VOICE = _require_kokoro()

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' cannot be empty")

    voice = (req.voice or DEFAULT_VOICE).strip()
    if voice not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")

    try:
        output_path, gen_time = generate_audio(
            text=text, voice=voice, speed=req.speed or 1.0
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))

    # Build public URL under /audio
    # Ensure path normalization within the mounted directory
    base_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "tts", "output"
    )
    try:
        rel_name = os.path.relpath(output_path, base_output_dir)
    except Exception:
        rel_name = os.path.basename(output_path)
    url = f"/audio/{rel_name}"
    return TTSResponse(
        url=url, path=output_path, voice=voice, time_seconds=float(gen_time)
    )


# ------------------ Embeddings API ------------------


class LoadEmbeddingModelRequest(BaseModel):
    model_name: Optional[str] = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face model id for the embedding model.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Optional local cache directory for model files.",
    )


class LoadEmbeddingModelResponse(BaseModel):
    status: str
    loaded: bool
    model_name: Optional[str]


def _require_sentence_transformers():
    try:
        # Imported lazily to keep optional dependency boundary
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=f"sentence-transformers is required for embeddings: {exc}",
        )


@app.post(
    "/embeddings/load",
    response_model=LoadEmbeddingModelResponse,
    summary="Load an embedding model",
    description="Load a sentence-transformers embedding model for vector generation.",
    tags=["embeddings"],
)
def load_embedding_model(req: LoadEmbeddingModelRequest) -> LoadEmbeddingModelResponse:
    global _emb_model, _emb_model_name
    with _emb_lock:
        SentenceTransformer = _require_sentence_transformers()
        try:
            kwargs: Dict[str, Any] = {}
            if req.cache_dir:
                kwargs["cache_folder"] = req.cache_dir
            # Unload previous model implicitly by dropping reference
            _emb_model = SentenceTransformer(
                req.model_name or "sentence-transformers/all-MiniLM-L6-v2", **kwargs
            )
            _emb_model_name = req.model_name or "sentence-transformers/all-MiniLM-L6-v2"
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return LoadEmbeddingModelResponse(
            status="loaded", loaded=True, model_name=_emb_model_name
        )


class UnloadEmbeddingResponse(BaseModel):
    status: str
    unloaded: bool


@app.post(
    "/embeddings/unload",
    response_model=UnloadEmbeddingResponse,
    summary="Unload the embedding model",
    description="Unload the currently loaded embedding model and free resources.",
    tags=["embeddings"],
)
def unload_embedding_model() -> UnloadEmbeddingResponse:
    global _emb_model, _emb_model_name
    with _emb_lock:
        was_loaded = _emb_model is not None
        _emb_model = None
        _emb_model_name = None
        gc.collect()
        return UnloadEmbeddingResponse(status="unloaded", unloaded=was_loaded)


class EmbeddingStatusResponse(BaseModel):
    loaded: bool
    model_name: Optional[str]


@app.get(
    "/embeddings/status",
    response_model=EmbeddingStatusResponse,
    summary="Embedding model status",
    description="Return whether an embedding model is loaded and its model id.",
    tags=["embeddings"],
)
def embeddings_status() -> EmbeddingStatusResponse:
    return EmbeddingStatusResponse(
        loaded=_emb_model is not None, model_name=_emb_model_name
    )


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of input strings to embed.")
    normalize: bool = Field(default=True, description="L2-normalize output vectors.")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int


@app.post(
    "/embeddings/generate",
    response_model=EmbedResponse,
    summary="Generate embeddings",
    description="Generate embeddings for a list of input texts using the loaded embedding model.",
    tags=["embeddings"],
    responses={
        503: {"description": "No embedding model is currently loaded."},
    },
)
def generate_embeddings(req: EmbedRequest) -> EmbedResponse:
    if _emb_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded yet")
    try:
        vectors = _emb_model.encode(req.texts)  # type: ignore[attr-defined]
        vectors = np.array(vectors, dtype=np.float32)
        if req.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return EmbedResponse(embeddings=vectors.tolist(), dim=int(vectors.shape[1]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SimilarityRequest(BaseModel):
    a: List[float] = Field(..., description="First embedding vector")
    b: List[float] = Field(..., description="Second embedding vector")
    metric: Literal["cosine", "dot"] = Field(default="cosine")
    normalize: bool = Field(
        default=True,
        description="If true, L2-normalize inputs before computing similarity.",
    )


class SimilarityResponse(BaseModel):
    similarity: float


@app.post(
    "/embeddings/similarity",
    response_model=SimilarityResponse,
    summary="Compute similarity",
    description="Compute similarity between two embedding vectors using cosine or dot product.",
    tags=["embeddings"],
)
def compute_similarity(req: SimilarityRequest) -> SimilarityResponse:
    a = np.array(req.a, dtype=np.float32)
    b = np.array(req.b, dtype=np.float32)
    if a.shape != b.shape:
        raise HTTPException(
            status_code=400, detail="Vectors 'a' and 'b' must have the same shape"
        )
    if req.normalize:
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an > 0:
            a = a / an
        if bn > 0:
            b = b / bn
    if req.metric == "cosine":
        sim = float(np.dot(a, b))
    else:
        sim = float(np.dot(a, b))
    return SimilarityResponse(similarity=sim)
