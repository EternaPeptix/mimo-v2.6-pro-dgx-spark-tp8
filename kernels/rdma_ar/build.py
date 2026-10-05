#!/usr/bin/env python3
"""build.py: compile rdma_ar.cu into <this dir>/build/rdma_ar_ext*.so for sm_121 (run inside the
SGLang image with the host's /usr/include/infiniband mounted)."""
import os

from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
load(
    name="rdma_ar_ext",
    sources=[os.path.join(here, "rdma_ar.cu")],
    build_directory=build,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    extra_ldflags=["-L/usr/lib/aarch64-linux-gnu", "-l:libibverbs.so.1"],
    verbose=False,
)
print("BUILD_OK", sorted(f for f in os.listdir(build) if f.endswith(".so")))
