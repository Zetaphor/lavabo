## Examples Guide

This directory contains Node.js example clients for the AI Toolkit API. Each script demonstrates a specific API capability.

### Prerequisites
- **Node.js**: v18+ (uses global `fetch`)
- **API server**: running at `http://localhost:8000` by default (see `server/README.md` for setup)

### Configuration
- **Base URL**: Set `AI_TOOLKIT_BASE_URL` to point at your API if not on the default
  - Example: `export AI_TOOLKIT_BASE_URL=http://localhost:8000`

### Running examples
From the repository root:

```bash
node examples/chat_basic.mjs
node examples/chat_structured.mjs
node examples/embeddings.mjs
node examples/kokoro_tts.mjs
node examples/piper_tts.mjs
node examples/clip_classify.mjs
node examples/moondream.mjs
node examples/vram.mjs
```

---

### chat_basic.mjs — Basic chat completion
- **What it does**:
  - Health check, load a GGUF model from Hugging Face (`unsloth/Qwen3-1.7B-GGUF`),
  - Send a simple chat prompt,
  - Print the assistant response,
  - Unload the model.
- **Key endpoints**: `GET /healthz`, `POST /load_gguf`, `POST /chat`, `POST /unload_gguf`
- **Customization**: Adjust `hf_repo`, `hf_file`, `n_ctx`, and `chat_format` in the script for your model.

Run:
```bash
node examples/chat_basic.mjs
```

Expected output includes the model load status and a short answer (e.g., a haiku) from the model.

---

### chat_structured.mjs — Structured JSON output
- **What it does**:
  - Loads a GGUF model,
  - Sends a prompt with `response_format: { type: 'json_object', schema }` to enforce JSON output,
  - Prints a parsed JSON object (falls back to manual `JSON.parse` if needed),
  - Unloads the model.
- **Key endpoint**: `POST /chat` with `response_format`
- **Note**: The server also returns a `parsed` field when decoding succeeds.

Run:
```bash
node examples/chat_structured.mjs
```

Expected output shows a JSON object matching the schema (fields like `affected_attribute`, `amount`, etc.).

---

### embeddings.mjs — Embedding generation and similarity
- **What it does**:
  - Loads a sentence-transformers model,
  - Generates embeddings for a few texts (optionally normalized),
  - Computes similarity between two vectors,
  - Unloads the embedding model.
- **Key endpoints**:
  - `POST /embeddings/load`
  - `POST /embeddings/generate`
  - `POST /embeddings/similarity`
  - `POST /embeddings/unload`
- **Prerequisite**: The API server must have `sentence-transformers` available (handled in the server container).

Run:
```bash
node examples/embeddings.mjs
```

Expected output includes the embedding dimension and a similarity score (e.g., cosine similarity ~0.5–0.9 depending on inputs).

---

### clip_classify.mjs — CLIP zero-shot image classification
- **What it does**:
  - Loads a CLIP model (`openai/clip-vit-base-patch32`),
  - Classifies a local image against candidate labels using zero-shot classification,
  - Performs an NSFW check (label "nsfw content" vs threshold),
  - Unloads the CLIP model.
- **Key endpoints**:
  - `POST /clip/load`
  - `POST /clip/classify`
  - `POST /clip/nsfw`
  - `POST /clip/unload`
- **Run**:
```bash
CLIP_IMAGE=/absolute/path/to/image.jpg node examples/clip_classify.mjs
```

---

### kokoro_tts.mjs — Text-to-speech (Kokoro)
- **What it does**:
  - Lists available TTS voices,
  - Synthesizes speech for a given text,
  - Prints a URL to the generated WAV under `/audio/...`.
- **Key endpoints**: `GET /kokoro/voices`, `POST /kokoro/synthesize`
- **Prerequisite**: Kokoro TTS must be installed and configured in the server (`server/tts`). If unavailable, the API returns an error.
- **Playback**: Use a browser to open the printed URL or play via ffmpeg:

```bash
node examples/kokoro_tts.mjs
```

Run:
```bash
node examples/kokoro_tts.mjs
```

Expected output shows available voices and a public URL to the generated audio.

---

### piper_tts.mjs — Text-to-speech (Piper)
- **What it does**:
  - Lists available Piper voices,
  - Optionally downloads a requested voice from Hugging Face (`rhasspy/piper-voices`),
  - Synthesizes speech with the selected voice,
  - Prints a URL to the generated WAV under `/audio/...`.
- **Key endpoints**: `GET /piper/voices`, `POST /piper/download`, `POST /piper/synthesize`
- **Prerequisites**:
  - The server must have Piper installed (`piper-tts` Python package).
  - Voices must be present under `/models/piper` or a directory listed in `PIPER_VOICES_DIR`.
  - This example can auto-download a voice from the Hugging Face catalog if missing.
- **Voice catalog**: [`rhasspy/piper-voices` on Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main)

Run:
```bash
node examples/piper_tts.mjs
```

Notes:
- The script requests `en_US-hfc_female-medium` by default. Change `requestedVoice` in `examples/piper_tts.mjs` to pick another, or set it to an empty value to use the default/local voice.
- Playback example:
  ```bash
  ffplay -autoexit -nodisp $(node -e "console.log(require('./examples/util.mjs').default.BASE_URL)")/audio/<printed-filename>.wav
  ```

---

### moondream.mjs — Vision-language (caption, VQA, detect, point)
- **What it does**:
  - Loads the Moondream VLM (`moondream/moondream-2b-2025-04-14-4bit`),
  - Generates a short and normal caption for a local image,
  - Asks a visual question (VQA),
  - Runs simple object detection and pointing,
  - Unloads the model.
- **Key endpoints**:
  - `POST /moondream/load`
  - `POST /moondream/caption`
  - `POST /moondream/query`
  - `POST /moondream/detect`
  - `POST /moondream/point`
  - `POST /moondream/unload`
- **Run**:
```bash
MOONDREAM_IMAGE=/absolute/path/to/image.jpg node examples/moondream.mjs
```
Note: The server uses Hugging Face `transformers` with `trust_remote_code` for Moondream. For GPU, ensure CUDA is available; else it falls back to CPU.

References:
- Moondream 4-bit model card: [Hugging Face](https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit)
- Official docs and recipes: [moondream.ai](https://moondream.ai/c/docs/introduction?utm_source=openai)

---

### vram.mjs — GPU VRAM usage
- **What it does**:
  - Queries VRAM statistics and prints total/used/free per GPU and memory used by the API process.
- **Key endpoint**: `GET /vram`
- **Prerequisite**: NVIDIA drivers, NVML available in the server container; otherwise the script reports that GPU/NVML is not available.

Run:
```bash
node examples/vram.mjs
```

Expected output lists GPUs and memory figures, or a notice if unavailable.

---

### util.mjs — Helper utilities
This is a small helper module used by the examples:
- `BASE_URL`: reads from `AI_TOOLKIT_BASE_URL` or defaults to `http://localhost:8000`
- `postJSON(path, payload)`: POST JSON and parse the response
- `getJSON(path)`: GET JSON and parse the response
- `prettyBytes(num)`: human-readable byte formatting (used by `vram.mjs`)

---

### Troubleshooting
- **Connection refused**: Ensure the API server is running and `AI_TOOLKIT_BASE_URL` points to it.
- **Model not loaded / 503**: Load a model before calling `/chat`, or use the example scripts which load/unload automatically.
- **Embeddings 500**: Ensure `sentence-transformers` is available in the server environment.
- **Kokoro TTS 500**: Kokoro TTS not installed/configured; see `server/tts`.
- **Piper TTS 500**: Ensure `piper-tts` is installed on the server and that voices exist under `/models/piper` or `PIPER_VOICES_DIR`. You can also call `POST /piper/download` from the example to fetch a voice.
- **VRAM empty**: Ensure GPUs are visible in the container (`--gpus all`) and NVML is present.


