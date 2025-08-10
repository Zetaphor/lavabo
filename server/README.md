## Lavabo Docker API — Quick Test Guide

This guide shows how to build and test the FastAPI llama.cpp server running inside Docker.

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU + drivers (for CUDA)
- NVIDIA Container Toolkit installed and configured
- GGUF model files on host under `/home/zetaphor/LLMs` (mounted to `/models` in the container)

### Build and run
- Build:
  ```bash
  docker compose -f server/docker-compose.yml build
  ```
- Run:
  ```bash
  docker compose -f server/docker-compose.yml up
  ```
- The API will listen on `http://localhost:8000`

Notes:
- `server/docker-compose.yml` maps host models dir `/home/zetaphor/LLMs` to container `/models`
- Environment:
  - `GGML_CUDA=1` enables CUDA backend
  - `N_GPU_LAYERS=-1` offloads all layers if possible

### Explore the API
- Open `http://localhost:8000/docs` for interactive Swagger UI to try endpoints.
- Alternative docs: `http://localhost:8000/redoc`
- OpenAPI schema: `http://localhost:8000/openapi.json`

### Endpoints
- `GET /healthz` → health check
- `GET /status` → model load status and config
- `GET /vram` → GPU VRAM usage (per device and by this process)
- `POST /load_gguf` → load a model (by absolute file path or Hugging Face repo + filename)
- `POST /unload_gguf` → unload current model
- `POST /chat` → chat completion with the loaded model
- `POST /embeddings/load` → load an embedding model (sentence-transformers)
- `POST /embeddings/unload` → unload the embedding model
- `GET /embeddings/status` → embedding model status
- `POST /embeddings/generate` → generate embeddings for input texts
- `POST /embeddings/similarity` → compute similarity between two vectors
- `POST /clip/load` → load a CLIP model
- `GET /clip/status` → CLIP model status and device
- `POST /clip/classify` → zero-shot classify an image (base64) against labels
- `POST /clip/nsfw` → NSFW check using CLIP
- `POST /clip/unload` → unload CLIP
- `POST /moondream/load` → load Moondream VLM (HF: `moondream/moondream-2b-2025-04-14-4bit`)
- `GET /moondream/status` → Moondream status and device
- `POST /moondream/caption` → caption an image
- `POST /moondream/query` → visual QA for an image
- `POST /moondream/detect` → text-queried object detection
- `POST /moondream/point` → pointing (visual grounding)
- `POST /moondream/unload` → unload Moondream
- `GET /kokoro/voices` → list Kokoro voices
- `POST /kokoro/synthesize` → synthesize speech with Kokoro
- `GET /piper/voices` → list Piper voices
- `POST /piper/synthesize` → synthesize speech with Piper
- `POST /piper/download` → download a Piper voice from Hugging Face
- `GET /audio/...` → static files for generated TTS audio
- `GET /docs` → interactive API docs (Swagger UI)
- `GET /redoc` → alternative API docs (ReDoc)
- `GET /openapi.json` → OpenAPI schema

#### Kokoro TTS

- `GET /kokoro/voices` → list available voices
- `POST /kokoro/synthesize` → synthesize speech. Body:
  ```json
  { "text": "Hello there", "voice": "af_heart", "speed": 1.0 }
  ```
  Response:
  ```json
  { "url": "/audio/output_1739907610.wav", "path": "/app/server/tts/output/output_1739907610.wav", "voice": "af_heart", "time_seconds": 0.85 }
  ```
  Download the file from the returned `url` on the same host where the API is served.

#### Piper TTS

- Ensure Piper models are available. Place `.onnx` and matching `.json` files under one of:
  - `/models/piper` (host-mounted via compose)
  - a custom directory set via `PIPER_VOICES_DIR` (colon-separated allowed)
- Optional env vars:
  - `PIPER_DEFAULT_VOICE` to set the default voice base name
  - Note: CUDA usage is auto-selected by onnxruntime if available; no explicit flag is required.

- `GET /piper/voices` → list available voices and the default voice
- `POST /piper/download` → download a voice by id from the Rhasspy Piper voices repo. Body:
  ```json
  { "voice": "en_US-ryan-high" }
  ```
  Voice catalog: [rhasspy/piper-voices on Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main)
- `POST /piper/synthesize` → synthesize speech. Body:
  ```json
  { "text": "Hello there", "voice": "en_US-ryan-high" }
  ```
  Note: Additional fields like `length_scale`, `noise_scale`, `noise_w`, `speaker_id`, and
  `sentence_silence` may be accepted by the schema but are currently ignored by the server.
  Response:
  ```json
  { "url": "/audio/piper_en_US-ryan-high_1739907610.wav", "path": "/app/server/tts/output/piper_en_US-ryan-high_1739907610.wav", "voice": "en_US-ryan-high", "time_seconds": 0.52 }
  ```
  Download the file from the returned `url` on the same host where the API is served.

### Load a model
- By Hugging Face repo id + filename:
  ```bash
  curl -X POST http://localhost:8000/load_gguf \
    -H "Content-Type: application/json" \
    -d '{
      "hf_repo": "unsloth/Qwen3-1.7B-GGUF",
      "hf_file": "Qwen3-1.7B-Q8_0.gguf",
      "n_ctx": 4096,
      "chat_format": "chatml"
    }'
  ```
  Notes:
  - Requires the `huggingface_hub` package in the server image. If missing, the API returns a 500 with an explanatory error.
  - Files are downloaded under `MODELS_DIR` (defaults to `/models`).

- By explicit GGUF file path (inside container):
  ```bash
  curl -X POST http://localhost:8000/load_gguf \
    -H "Content-Type: application/json" \
    -d '{
      "file": "/models/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/qwen2.5-coder-3b-instruct-q4_0.gguf",
      "n_ctx": 4096,
      "n_gpu_layers": -1,
      "chat_format": "chatml"
    }'
  ```
  The `file` path must exist inside the container filesystem.

- Check status:
  ```bash
  curl http://localhost:8000/status
  ```

### Chat completion
- Basic:
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Write a haiku about GPUs."}
      ],
      "max_tokens": 128,
      "temperature": 0.7
    }'
  ```

- Enforce JSON output via schema:
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "system", "content": "Return only valid minified JSON."},
        {"role": "user", "content": "John realizes he is in a simulation."}
      ],
      "response_format": {
        "type": "json_object",
        "schema": {
          "type":"object",
          "properties": {
            "affected_attribute":{"type":"string"},
            "amount":{"type":"number"},
            "mood":{"type":"string"},
            "event_description":{"type":"string"},
            "inner_thoughts":{"type":"string"}
          },
          "required":["affected_attribute","amount","mood","event_description","inner_thoughts"]
        }
      },
      "temperature": 0.2
    }'
  ```

  The response body includes:
  - `content`: the model's raw string output
  - `parsed`: when `response_format.type == "json_object"`, the server attempts to JSON-decode `content` and returns the resulting object (or `null` on failure)
  - `raw`: the full underlying llama.cpp response

### Unload model
```bash
curl -X POST http://localhost:8000/unload_gguf
```

### VRAM usage
```bash
curl http://localhost:8000/vram
```
- Requires NVML (`pynvml`) and NVIDIA toolkit/drivers. Shows total/free per GPU and memory used by the API process.
- Node example: `node examples/vram.mjs`

### Health
```bash
curl http://localhost:8000/healthz
```

### Embeddings

- Load embedding model:
  ```bash
  curl -X POST http://localhost:8000/embeddings/load \
    -H "Content-Type: application/json" \
    -d '{
      "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }'
  ```

- Check status:
  ```bash
  curl http://localhost:8000/embeddings/status
  ```

- Generate embeddings:
  ```bash
  curl -X POST http://localhost:8000/embeddings/generate \
    -H "Content-Type: application/json" \
    -d '{
      "texts": ["what is the weather today?", "launch the music player"],
      "normalize": true
    }'
  ```

- Compute similarity:
  ```bash
  curl -X POST http://localhost:8000/embeddings/similarity \
    -H "Content-Type: application/json" \
    -d '{
      "a": [0.1, 0.2, 0.3],
      "b": [0.1, 0.2, 0.25],
      "metric": "cosine",
      "normalize": true
    }'
  ```

- Unload embedding model:
  ```bash
  curl -X POST http://localhost:8000/embeddings/unload
  ```

### CLIP

- Load CLIP:
  ```bash
  curl -X POST http://localhost:8000/clip/load \
    -H "Content-Type: application/json" \
    -d '{
      "model_name": "openai/clip-vit-base-patch32",
      "device": "auto"
    }'
  ```

- Status:
  ```bash
  curl http://localhost:8000/clip/status
  ```

- Classify image (base64):
  ```bash
  IMG_B64=$(base64 -w0 /path/to/image.jpg)
  curl -X POST http://localhost:8000/clip/classify \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\": \"$IMG_B64\", \"labels\": [\"safe for work content\", \"nsfw content\", \"a photo of a person\"]}"
  ```

- NSFW check (base64):
  ```bash
  IMG_B64=$(base64 -w0 /path/to/image.jpg)
  curl -X POST http://localhost:8000/clip/nsfw \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\": \"$IMG_B64\", \"threshold\": 0.5}"
  ```

- Unload CLIP:
  ```bash
  curl -X POST http://localhost:8000/clip/unload
  ```

### Moondream VLM

- Load Moondream (auto-select CUDA if available):
  ```bash
  curl -X POST http://localhost:8000/moondream/load \
    -H "Content-Type: application/json" \
    -d '{
      "model_name": "moondream/moondream-2b-2025-04-14-4bit",
      "device": "auto",
      "compile": true
    }'
  ```

- Force GPU usage (if CUDA is available in the container):
  ```bash
  curl -X POST http://localhost:8000/moondream/load \
    -H "Content-Type: application/json" \
    -d '{
      "model_name": "moondream/moondream-2b-2025-04-14-4bit",
      "device": "cuda",
      "compile": true
    }'
  ```

Notes:
- Ensure NVIDIA Container Toolkit is installed and GPUs are visible to Docker.
- The provided `docker-compose.yml` already requests GPU via `deploy.resources.reservations.devices`.
- The Dockerfile is configured to use CUDA wheels (`PIP_EXTRA_INDEX_URL` set to cu121).

- Status:
  ```bash
  curl http://localhost:8000/moondream/status
  ```

- Caption:
  ```bash
  IMG_B64=$(base64 -w0 /path/to/image.jpg)
  curl -X POST http://localhost:8000/moondream/caption \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\": \"$IMG_B64\", \"length\": \"normal\"}"
  ```

- Visual query (VQA):
  ```bash
  IMG_B64=$(base64 -w0 /path/to/image.jpg)
  curl -X POST http://localhost:8000/moondream/query \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\": \"$IMG_B64\", \"question\": \"How many people are in the image?\"}"
  ```

- Detect objects by text query:
  ```bash
  IMG_B64=$(base64 -w0 /path/to/image.jpg)
  curl -X POST http://localhost:8000/moondream/detect \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\": \"$IMG_B64\", \"query\": \"face\"}"
  ```

- Pointing (visual grounding):
  ```bash
  IMG_B64=$(base64 -w0 /path/to/image.jpg)
  curl -X POST http://localhost:8000/moondream/point \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\": \"$IMG_B64\", \"query\": \"person\"}"
  ```

- Unload:
  ```bash
  curl -X POST http://localhost:8000/moondream/unload
  ```

References:
- Model card (4-bit): https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit
- Official docs and recipes: https://moondream.ai/c/docs/introduction

### Troubleshooting
- Build fails with CMake option errors: we use `GGML_CUDA=on` (new llama.cpp option). Ensure CUDA toolchain and NVIDIA Container Toolkit are installed.
- Model not found: verify your GGUF path exists inside the container under `/models` and that your host files are in `/home/zetaphor/LLMs`.
- Out of memory: try lowering `n_gpu_layers` or `n_ctx`.
- VRAM endpoint empty: ensure GPUs are visible in the container (`--gpus all` via compose) and NVML is available.


