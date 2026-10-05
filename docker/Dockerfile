# MiMo-V2.6-Pro-MOPD across 8x DGX Spark (GB10, sm_121): stock SGLang v0.5.21 + GB10 patches + custom kernels.
#
# Build from the repository root (no GPU needed; every node needs the image):
#   docker build -f docker/Dockerfile -t mimo-gb10:v0.5.21 .
FROM lmsysorg/sglang:v0.5.21-cu130@sha256:d6ae3ad037c65436365b7c4fd1a20a6f90d5ce478bbc4d6618c14365bb1d1822

# MiMo-V2's model module imports its multimodal (audio) processor even when serving text only;
# without torchcodec, startup fails with "No processor registered for architecture: MiMoV2ForCausalLM".
RUN pip install --no-cache-dir --no-deps torchcodec==0.17.0

# Headers for the RDMA all-reduce extension (libibverbs/libmlx5 runtime libraries are already present).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# SGLang source patches (see docs/OPTIMIZATIONS.md for what each one does).
COPY patches/sglang-v0.5.21 /opt/mimo-gb10/patches
RUN cd /sgl-workspace/sglang \
    && for p in /opt/mimo-gb10/patches/*.patch; do \
         echo "applying $(basename "$p")" && patch -p1 --forward --no-backup-if-mismatch < "$p" || exit 1; \
       done

# Custom CUDA/C++ extensions, compiled for GB10 at build time.
COPY kernels/rdma_ar /opt/mimo-gb10/rdma_ar
COPY kernels/skinny_gemm /opt/mimo-gb10/skinny_gemm
ENV TORCH_CUDA_ARCH_LIST=12.1
RUN cd /opt/mimo-gb10/rdma_ar && python3 build.py \
    && cd /opt/mimo-gb10/skinny_gemm && python3 build.py

ENV RDMA_AR_DIR=/opt/mimo-gb10/rdma_ar \
    RDMA_AR_BUILD_DIR=/opt/mimo-gb10/rdma_ar/build \
    SKINNY_GEMM_DIR=/opt/mimo-gb10/skinny_gemm/build

# Tests and micro-benchmarks for the kernels ship in the image too.
COPY kernels/tests /opt/mimo-gb10/tests
