#!/usr/bin/env python3
"""build.py: compile skinny_gemm.cu into <this dir>/build/skinny_gemm_ext*.so for sm_121."""
import os
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build"); os.makedirs(build, exist_ok=True)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
load(name="skinny_gemm_ext", sources=[os.path.join(here, "skinny_gemm.cu")], build_directory=build,
     extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
print("BUILD_OK", sorted(f for f in os.listdir(build) if f.endswith(".so")))
