ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to perform system updates and install packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install Python dependencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build scikit-build-core ninja wheel setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context pynvml huggingface_hub

# Install llama-cpp-python (build with CUDA via GGML)
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-build-isolation --force-reinstall --no-cache-dir llama-cpp-python

# Default command runs the API via uvicorn, but can be overridden by compose
ENV HOST=0.0.0.0 PORT=8000
CMD ["uvicorn", "server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

WORKDIR /app

