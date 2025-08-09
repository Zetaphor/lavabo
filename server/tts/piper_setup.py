# pyright: reportMissingTypeStubs=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile  # type: ignore


def _get_default_voice_dirs() -> List[str]:
    env_dirs = os.environ.get("PIPER_VOICES_DIR", "").strip()
    dirs: List[str] = []
    if env_dirs:
        # Allow colon or comma separated
        for part in re.split(r"[:,]", env_dirs):
            part = part.strip()
            if part:
                dirs.append(part)
    # Fallbacks: host-mounted models dir and a local repo path
    dirs.extend(
        [
            "/models/piper",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper_voices"),
        ]
    )
    # Deduplicate while preserving order
    seen = set()
    unique_dirs: List[str] = []
    for d in dirs:
        if d not in seen:
            unique_dirs.append(d)
            seen.add(d)
    return unique_dirs


def _discover_voice_files(
    voice_dirs: Optional[List[str]] = None,
) -> Dict[str, Tuple[str, Optional[str]]]:
    voice_dirs = voice_dirs or _get_default_voice_dirs()
    discovered: Dict[str, Tuple[str, Optional[str]]] = {}
    for base in voice_dirs:
        if not base or not os.path.isdir(base):
            continue
        # Recursively search for voice files
        onnx_map: Dict[str, str] = {}
        json_map: Dict[str, str] = {}
        for root, _dirs, files in os.walk(base):
            for fname in files:
                fpath = os.path.join(root, fname)
                if fname.endswith(".onnx"):
                    key = fname[:-5]
                    onnx_map[key] = fpath
                elif fname.endswith(".onnx.json"):
                    key = fname[:-10]
                    json_map[key] = fpath
                elif fname.endswith(".json"):
                    # Fallback for non-standard naming
                    key = fname[:-5]
                    json_map.setdefault(key, fpath)
        for base_name, onnx_path in onnx_map.items():
            # Prefer matching .onnx.json; fallback to plain .json; accept onnx-only
            cfg_path = json_map.get(base_name)
            if base_name not in discovered:
                discovered[base_name] = (onnx_path, cfg_path)
    return discovered


# Lazy import and cache of PiperVoice instances
_VOICE_CACHE: Dict[str, Any] = {}


def list_voices() -> List[str]:
    files = _discover_voice_files()
    return sorted(list(files.keys()))


def get_default_voice() -> str:
    env_voice = (os.environ.get("PIPER_DEFAULT_VOICE", "") or "").strip()
    if env_voice:
        return env_voice
    voices = list_voices()
    if not voices:
        raise RuntimeError(
            "No Piper voices found. Place model .onnx and matching .json files under '/models/piper' or set PIPER_VOICES_DIR."
        )
    return voices[0]


def _load_voice(voice_name: str):
    """Load PiperVoice for name, caching the instance."""
    try:
        # Prefer top-level import; some builds export PiperVoice here
        from piper import PiperVoice  # type: ignore
    except Exception as exc:  # pragma: no cover
        try:
            # Fallback for alternate module structure
            from piper.voice import PiperVoice  # type: ignore
        except Exception:
            raise RuntimeError(f"piper-tts is required for Piper TTS: {exc}")

    if voice_name in _VOICE_CACHE:
        return _VOICE_CACHE[voice_name]

    files = _discover_voice_files()
    paths = files.get(voice_name)
    if not paths:
        raise ValueError(f"Piper voice not found: {voice_name}")
    onnx_path, _json_path = paths

    # Note: CUDA flag intentionally unused per Piper1 API that loads from ONNX CPU/GPU auto-handled by onnxruntime

    # Per Piper1 GPL docs: load from ONNX path only
    try:
        voice = PiperVoice.load(onnx_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Piper voice '{voice_name}': {exc}")
    _VOICE_CACHE[voice_name] = voice
    return voice


def _sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return value[:128]


def generate_audio_piper(
    text: str,
    voice: Optional[str] = None,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w: float = 0.8,
    speaker_id: Optional[int] = None,
    sentence_silence: float = 0.2,
) -> Tuple[str, float]:
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty")

    voice_name = (voice or get_default_voice()).strip()
    start_time = time.time()

    piper_voice = _load_voice(voice_name)
    # 1) Streaming iterator (per docs/tests)
    audio_chunks: List[np.ndarray] = []
    synth_iter = piper_voice.synthesize(text)
    for chunk in synth_iter:  # one chunk per sentence
        # chunk.audio_float_array is expected per tests
        data = getattr(chunk, "audio_float_array", None)
        if data is None:
            continue
        arr = np.asarray(data, dtype=np.float32)
        if arr.size:
            audio_chunks.append(arr)
    if not audio_chunks:
        raise RuntimeError("Piper synthesis returned no audio")

    wav: np.ndarray = np.concatenate(audio_chunks)

    # Derive sample rate from the first chunk or from voice.config
    first_chunk_rate = None
    try:
        first_chunk_rate = int(
            getattr(next(piper_voice.synthesize(".")), "sample_rate", 0)
        )
    except Exception:
        first_chunk_rate = None
    if not first_chunk_rate:
        cfg = getattr(piper_voice, "config", None)
        first_chunk_rate = int(getattr(cfg, "sample_rate", 22050))
    sample_rate = first_chunk_rate or 22050

    # Output directory shared with Kokoro so static /audio mount works
    module_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(module_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    ts = int(time.time())
    fname = f"piper_{_sanitize_filename(voice_name)}_{ts}.wav"
    out_path = os.path.join(output_dir, fname)

    # Convert to int16 PCM for compatibility
    wav_clipped = np.clip(wav, -1.0, 1.0)
    wav_int16 = (wav_clipped * 32767.0).astype(np.int16)
    wavfile.write(out_path, rate=int(sample_rate), data=wav_int16)

    return out_path, float(time.time() - start_time)


def _resolve_dest_dir(explicit_dest: Optional[str]) -> str:
    if explicit_dest and explicit_dest.strip():
        return explicit_dest.strip()
    # Prefer first entry from PIPER_VOICES_DIR if set
    env_dirs = os.environ.get("PIPER_VOICES_DIR", "").strip()
    if env_dirs:
        first = re.split(r"[:,]", env_dirs)[0].strip()
        if first:
            return first
    # Fallback to /models/piper
    return "/models/piper"


def download_voice_from_hf(
    voice: str, dest_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Download a Piper voice pair (.onnx and .onnx.json) from Hugging Face
    repo 'rhasspy/piper-voices' into dest_dir. Returns paths.

    Strategy: list files in repo and locate the two files whose basenames
    match the requested voice, e.g. 'en_US-ryan-high.onnx' and
    'en_US-ryan-high.onnx.json', regardless of their subdirectory.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"huggingface_hub is required to download Piper voices: {exc}"
        )

    api = HfApi()
    repo_id = "rhasspy/piper-voices"
    try:
        files = api.list_repo_files(repo_id=repo_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to list files from {repo_id}: {exc}")

    target_model_name = f"{voice}.onnx"
    target_config_name = f"{voice}.onnx.json"
    model_path_remote: Optional[str] = None
    config_path_remote: Optional[str] = None
    for f in files:
        if f.endswith(target_model_name):
            model_path_remote = f
        elif f.endswith(target_config_name):
            config_path_remote = f
        if model_path_remote and config_path_remote:
            break

    if not model_path_remote or not config_path_remote:
        raise FileNotFoundError(
            f"Voice '{voice}' not found in {repo_id}. See voices list: https://huggingface.co/{repo_id}/tree/main"
        )

    final_dir = _resolve_dest_dir(dest_dir)
    os.makedirs(final_dir, exist_ok=True)

    try:
        local_model = hf_hub_download(
            repo_id=repo_id,
            filename=model_path_remote,
            local_dir=final_dir,
            local_dir_use_symlinks=False,
        )
        local_config = hf_hub_download(
            repo_id=repo_id,
            filename=config_path_remote,
            local_dir=final_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download voice '{voice}': {exc}")

    return {
        "voice": voice,
        "model_path": local_model,
        "config_path": local_config,
        "dest_dir": final_dir,
    }
