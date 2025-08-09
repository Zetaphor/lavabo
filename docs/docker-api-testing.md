## AI Toolkit Docker API — Quick Test Guide

This guide shows how to build and test the FastAPI llama.cpp server running inside Docker.

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU + drivers (for CUDA)
- NVIDIA Container Toolkit installed and configured
- GGUF model files on host under `/home/zetaphor/LLMs` (mounted to `/models` in the container)

### Build and run
- Build:
  ```bash
  docker compose build
  ```
- Run:
  ```bash
  docker compose up
  ```
- The API will listen on `http://localhost:8000`

Notes:
- `docker-compose.yml` maps host models dir `/home/zetaphor/LLMs` to container `/models`
- Environment:
  - `GGML_CUDA=1` enables CUDA backend
  - `MODEL_NAME` default is `phi3` (you can change to `llama3`)
  - `N_GPU_LAYERS=-1` offloads all layers if possible

### Endpoints
- `GET /healthz` → health check
- `GET /status` → model load status and config
- `GET /vram` → GPU VRAM usage (per device and by this process)
- `POST /load_model` → load a model (by name or explicit file path)
- `POST /unload_model` → unload current model
- `POST /chat` → chat completion with the loaded model

### Load a model
- By predefined name:
  ```bash
  curl -X POST http://localhost:8000/load_model \
    -H "Content-Type: application/json" \
    -d '{"model_name": "phi3"}'
  ```
  Predefined: `phi3`, `llama3`.

- By explicit GGUF file path (inside container):
  ```bash
  curl -X POST http://localhost:8000/load_model \
    -H "Content-Type: application/json" \
    -d '{
      "file": "/models/microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
      "n_ctx": 4096,
      "n_gpu_layers": -1,
      "chat_format": "chatml"
    }'
  ```

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
      "temperature": 0.7
    }'
  ```

### Unload model
```bash
curl -X POST http://localhost:8000/unload_model
```

### VRAM usage
```bash
curl http://localhost:8000/vram
```
- Requires NVML (`pynvml`) and NVIDIA toolkit/drivers. Shows total/free per GPU and memory used by the API process.

### Health
```bash
curl http://localhost:8000/healthz
```

### Troubleshooting
- Build fails with CMake option errors: we use `GGML_CUDA=on` (new llama.cpp option). Ensure CUDA toolchain and NVIDIA Container Toolkit are installed.
- Model not found: verify your GGUF path exists inside the container under `/models` and that your host files are in `/home/zetaphor/LLMs`.
- Out of memory: try lowering `n_gpu_layers` or `n_ctx`.
- VRAM endpoint empty: ensure GPUs are visible in the container (`--gpus all` via compose) and NVML is available.


