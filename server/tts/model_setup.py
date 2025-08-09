import time
import re
import numpy as np
import os
import torch
from scipy.io import wavfile

# Ensure compatibility with environments where phonemizer's EspeakWrapper does not
# expose set_data_path by adding a no-op method before importing kokoro.
try:
    from phonemizer.backend.espeak import wrapper as _espeak_wrapper_mod  # type: ignore

    EspeakWrapper = getattr(_espeak_wrapper_mod, "EspeakWrapper", None)
    if EspeakWrapper is not None and not hasattr(EspeakWrapper, "set_data_path"):
        try:
            EspeakWrapper.set_data_path = classmethod(lambda cls, path: None)  # type: ignore[attr-defined]
        except Exception:
            pass
except Exception:
    pass

# Shim misaki API differences before importing kokoro
try:
    import misaki.en as _misaki_en  # type: ignore

    # Back-compat: some kokoro builds expect MutableToken
    if not hasattr(_misaki_en, "MutableToken") and hasattr(_misaki_en, "MToken"):
        try:
            _misaki_en.MutableToken = _misaki_en.MToken  # type: ignore[attr-defined]
        except Exception:
            pass

    # Back-compat: ensure expected token attributes are present
    try:
        _MToken = getattr(_misaki_en, "MToken", None)
        if _MToken is not None:
            # Provide attribute-like properties expected by older kokoro code paths
            if not hasattr(_MToken, "prespace"):
                try:
                    setattr(_MToken, "prespace", property(lambda self: ""))
                except Exception:
                    pass
            if not hasattr(_MToken, "postspace"):
                try:
                    setattr(_MToken, "postspace", property(lambda self: ""))
                except Exception:
                    pass
    except Exception:
        pass
except Exception:
    pass

from kokoro import KPipeline

# Get the directory where this module is located
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global pipeline setup (matches agent-test behavior)
device = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = KPipeline(lang_code="a", device=device)

# Available voices (matches agent-test)
AVAILABLE_VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]

# Default voice (matches agent-test)
DEFAULT_VOICE = "af_heart"


def resplit_strings(arr):
    if not arr:
        return "", ""
    if len(arr) == 1:
        return arr[0], ""

    min_diff = float("inf")
    best_split = 0
    lengths = [len(s) for s in arr]
    spaces = len(arr) - 1

    left_len = 0
    right_len = sum(lengths) + spaces
    for i in range(1, len(arr)):
        left_len += lengths[i - 1] + (1 if i > 1 else 0)
        right_len -= lengths[i - 1] + 1
        diff = abs(left_len - right_len)
        if diff < min_diff:
            min_diff = diff
            best_split = i

    return " ".join(arr[:best_split]), " ".join(arr[best_split:])


def recursive_split(text, max_tokens=510):
    if not text:
        return []
    tokens = text
    if len(tokens) < max_tokens:
        return [text] if text.strip() else []
    if " " not in text:
        return []

    for punctuation in ["!.?…", ":;", ",—"]:
        splits = re.split(
            f"(?:(?<=[{punctuation}])|(?<=[{punctuation}][\"'»])|(?<=[{punctuation}][\"'»][\"'»])) ",
            text,
        )
        if len(splits) > 1:
            break
    else:
        splits = text.split(" ")

    a, b = resplit_strings(splits)
    return recursive_split(a) + recursive_split(b)


def normalize_text(text, lang):
    text = re.sub(
        r"([a-zA-Z0-9-]+)\.([a-zA-Z0-9-]+)\.(com|net|org|edu|gov|io|ai)",
        r"\1 dot \2 dot \3",
        text,
    )
    text = re.sub(r"([a-zA-Z0-9-]+)\.(co|com)\.([a-z]{2})", r"\1 dot \2 dot \3", text)

    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')

    text = re.sub(r"<a[^>]*>([^<]+)</a>", r"A link to \1", text)

    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!\\)\\(?!\\)", " ", text)

    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=\n) +(?=\n)", "", text)

    return text.strip()


def clamp_speed(speed):
    if not isinstance(speed, (float, int)):
        return 1.0
    return max(0.5, min(2.0, float(speed)))


def generate_audio(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    newline_split: int = 2,
    skip_square_brackets: bool = True,
) -> tuple[str, float]:
    start_time = time.time()

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    output_dir = os.path.join(MODULE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    speed = clamp_speed(speed)

    if voice not in AVAILABLE_VOICES:
        raise ValueError(f"Unknown voice: {voice}")

    lang = voice[0]

    if skip_square_brackets:
        text = re.sub(r"\[.*?\]", "", text)

    text = normalize_text(text, lang)

    if not text.strip():
        raise ValueError("Text is empty after preprocessing")

    if newline_split > 0:
        texts = [t.strip() for t in re.split("\n{" + str(newline_split) + ",}", text)]
        texts = [t for t in texts if t]
    else:
        texts = [text]

    segments: list[str] = []
    for t in texts:
        segments.extend(s for s in recursive_split(t) if s.strip())

    if not segments:
        raise ValueError("No valid text segments to process after splitting")

    audio_segments = []
    for i, segment in enumerate(segments, 1):
        for _, _, audio in PIPELINE(segment, voice=voice, speed=speed):
            if audio is not None:
                audio_segments.append(audio)

    combined_audio = np.concatenate(audio_segments)

    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"output_{timestamp}.wav")
    wavfile.write(output_path, rate=24000, data=combined_audio)

    generation_time = time.time() - start_time
    return output_path, generation_time
