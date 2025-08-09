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

# Import FastAPI dynamically to avoid linter issues when type stubs are missing
try:  # pragma: no cover
    _fastapi = importlib.import_module("fastapi")  # type: ignore
    FastAPI = _fastapi.FastAPI  # type: ignore
    HTTPException = _fastapi.HTTPException  # type: ignore
except Exception:  # pragma: no cover
    _fastapi = None  # type: ignore

    class _HTTPException(Exception):
        pass

    FastAPI = object  # type: ignore
    HTTPException = _HTTPException  # type: ignore

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

app = FastAPI(title="AI Toolkit - llama.cpp API", version="0.2.0")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ResponseFormatSchema(TypedDict, total=False):
    type: Literal["json_object"]
    schema: Dict[str, Any]


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Ordered chat messages")
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    response_format: Optional[ResponseFormatSchema] = None


class ChatResponse(BaseModel):
    content: str
    raw: Optional[Dict[str, Any]] = None
    # If response_format.type == "json_object" and parsing succeeds
    parsed: Optional[Dict[str, Any]] = None


_llm: Optional[Llama] = None
_llm_lock = threading.Lock()
_loaded_cfg: Optional[ModelConfig] = None


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


class LoadModelResponse(BaseModel):
    status: str
    loaded: bool
    config: Dict[str, Any]


def _unload_llm_locked() -> None:
    global _llm, _loaded_cfg
    if _llm is not None:
        # Drop reference and force GC to release GPU memory
        _llm = None
        _loaded_cfg = None
        gc.collect()


@app.post("/load_model", response_model=LoadModelResponse)
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


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
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
            response_format=req.response_format,  # type: ignore[arg-type]
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=False,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))

    message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed: Optional[Dict[str, Any]] = None
    if req.response_format and req.response_format.get("type") == "json_object":
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


@app.post("/unload_model", response_model=UnloadResponse)
def unload_model() -> UnloadResponse:
    with _llm_lock:
        was_loaded = _llm is not None
        _unload_llm_locked()
        return UnloadResponse(status="unloaded", unloaded=was_loaded)


class StatusResponse(BaseModel):
    loaded: bool
    config: Optional[Dict[str, Any]]


@app.get("/status", response_model=StatusResponse)
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


@app.get("/vram", response_model=VRAMUsageResponse)
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
