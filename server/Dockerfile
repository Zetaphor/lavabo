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

# Install Python dependencies from pinned lockfile
ENV GIT_TERMINAL_PROMPT=0
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY server/requirements.lock.txt /app/server/requirements.lock.txt
RUN python3 -m pip install --upgrade pip
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN python3 -m pip install --no-cache-dir --no-deps -r /app/server/requirements.lock.txt

# Ensure the Piper library matches the documented API (OHF-Voice piper1-gpl)
# Reference docs/tests: https://github.com/OHF-Voice/piper1-gpl/blob/7c40f1b83c9725bb039a5d5d0e302adfd3ab530b/tests/test_piper.py
RUN python3 -m pip install --no-cache-dir --no-deps \
    git+https://github.com/OHF-Voice/piper1-gpl@7c40f1b83c9725bb039a5d5d0e302adfd3ab530b

# Ensure espeak-ng data is discoverable by phonemizer/espeak-ng loader
ENV ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data

# Ensure llama-cpp-python is compiled against CUDA with the same pinned version
RUN LCP_VER=$(python3 -c "import importlib.metadata as m; print(m.version('llama-cpp-python'))") \
    && pip install --no-deps --no-binary=llama-cpp-python --no-build-isolation --force-reinstall --no-cache-dir llama-cpp-python==${LCP_VER}

# Default command runs the API via uvicorn, but can be overridden by compose
ENV HOST=0.0.0.0 PORT=8000
CMD ["uvicorn", "server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

WORKDIR /app

