AI Toolkit
=========

A modular Python toolkit for building complex AI toolchains. The first module implements a GGUF model loader with optional structured outputs.

Features
--------

- Load models from Hugging Face repos or local file paths
- Perform inference with `llama-cpp-python`
- Optional structured output using Pydantic schemas

Quickstart (with uv)
--------------------

```bash
# Create and activate virtual env with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

GPU builds with uv
------------------

The loader defaults to GPU offload (`n_gpu_layers=-1`). To enable GPU support in `llama-cpp-python`, install it with the appropriate build flags for your platform before installing this package.

- NVIDIA CUDA (cuBLAS):
  ```bash
  # Requires CUDA toolkit/driver. Set archs if you know them (e.g., 86 for Ampere):
  export CMAKE_ARGS="-DLLAMA_CUBLAS=on"  # optionally: -DCMAKE_BUILD_TYPE=Release
  # Optional: export CUDAARCHS=86
  uv pip install --no-build-isolation --force-reinstall --no-cache-dir llama-cpp-python
  ```

- Apple Silicon / Metal:
  ```bash
  export CMAKE_ARGS="-DLLAMA_METAL=on"
  uv pip install --no-build-isolation --force-reinstall --no-cache-dir llama-cpp-python
  ```

- AMD ROCm (hipBLAS):
  ```bash
  export CMAKE_ARGS="-DLLAMA_HIPBLAS=on"
  # Optionally target specific GPUs, e.g.: export HSA_OVERRIDE_GFX_VERSION=11.0.0
  uv pip install --no-build-isolation --force-reinstall --no-cache-dir llama-cpp-python
  ```

- CPU with BLAS (OpenBLAS/Accelerate/etc.):
  ```bash
  export CMAKE_ARGS="-DLLAMA_BLAS=on -DLLAMA_BLAS_VENDOR=OpenBLAS"
  uv pip install --no-build-isolation --force-reinstall --no-cache-dir llama-cpp-python
  ```

After installing `llama-cpp-python` with the desired backend, install this package:
```bash
uv pip install -e .
```

Usage
-----

```python
from ai_toolkit.gguf.loader import GGUFModelLoader
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Load by Hugging Face repo (defaults to GPU offload if available)
loader = GGUFModelLoader.from_hf(
    repo_id="TheBloke/Llama-2-7B-GGUF",
    filename="llama-2-7b.Q4_K_M.gguf",
)

# Or load from local path
# loader = GGUFModelLoader.from_path("/path/to/model.gguf")

# Free-form generation
print(loader.generate("Write a short poem about summer."))

# Structured output
result = loader.generate("Extract the person's name and age from: 'Alice is 32 years old'", schema=Person)
print(result)  # -> Person(name='Alice', age=32)
```

Notes
-----

- By default `n_gpu_layers=-1` to use GPU offload across all layers (when compiled with CUDA/Metal/OpenCL). If the environment does not support GPU offload, the loader falls back to CPU with a warning. You can force CPU by passing `n_gpu_layers=0`.

License
-------

MIT


