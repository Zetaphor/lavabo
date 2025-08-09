ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to perform system updates and install packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev espeak-ng \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && ln -sf /usr/lib/x86_64-linux-gnu/espeak-ng-data /usr/share/espeak-ng-data || true

# Install Python dependencies
# Install CPU-only PyTorch for embeddings to avoid CUDA lib mismatches
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

# Core deps and embeddings stack
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build scikit-build-core ninja wheel setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context pynvml huggingface_hub sentence-transformers numpy scipy phonemizer==3.2.1 soundfile librosa kokoro==0.3.4
ENV GIT_TERMINAL_PROMPT=0

# Ensure espeak-ng data is discoverable by phonemizer/espeak-ng loader
ENV ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data

# Install llama-cpp-python (build with CUDA via GGML)
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-build-isolation --force-reinstall --no-cache-dir llama-cpp-python

# Default command runs the API via uvicorn, but can be overridden by compose
ENV HOST=0.0.0.0 PORT=8000
CMD ["uvicorn", "server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

WORKDIR /app

