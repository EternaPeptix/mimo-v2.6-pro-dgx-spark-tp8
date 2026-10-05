"""Python side of the one-shot RDMA all-reduce (see rdma_ar.cu).

RdmaAllReduce(cpu_group, rank, world, device) builds the QPs to every peer over the gloo
`cpu_group` and exposes should_use(x) / all_reduce(x) (out-of-place, bf16).

Env knobs:
  RDMA_AR_BUILD_DIR   directory holding rdma_ar_ext*.so          (default: <this dir>/build)
  RDMA_AR_HCAS        comma list of RoCE devices, pair (i,j) uses hcas[(i+j) % n]
                      (default rocep1s0f0,roceP2p1s0f1: one per PCIe link)
  RDMA_AR_MAX_BYTES   slot size = largest all-gather input handled (default 2 MiB)
  RDMA_AR_MAX_REDUCE_BYTES  largest all-reduce handled; above this NCCL's ring moves fewer bytes
                      than one-shot (default 512 KiB)
  RDMA_AR_TWOSHOT_MIN_BYTES  all-reduces of at least this size use two-shot (reduce-scatter +
                      all-gather, 4x fewer wire bytes, one extra hop); smaller ones use one-shot
                      (default 32 KiB; 0 disables two-shot)
  RDMA_AR_GID_INDEX   RoCEv2 GID index (default NCCL_IB_GID_INDEX or 5)
  RDMA_AR_PROXY_CORE  CPU core for the proxy thread (default -1 = no pinning)
  RDMA_AR_TIMEOUT_MS  GPU-side wait timeout before trapping (default 60000)
"""
import os
import sys

import torch
import torch.distributed as dist

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        build = os.environ.get(
            "RDMA_AR_BUILD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        )
        if build not in sys.path:
            sys.path.insert(0, build)
        import rdma_ar_ext  # noqa: E402

        _EXT = rdma_ar_ext
    return _EXT


class RdmaAllReduce:
    def __init__(self, cpu_group, rank: int, world: int, device: torch.device):
        ext = _load_ext()
        hcas = [h for h in os.environ.get("RDMA_AR_HCAS", "rocep1s0f0,roceP2p1s0f1").split(",") if h]
        self.max_bytes = int(os.environ.get("RDMA_AR_MAX_BYTES", str(2 * 1024 * 1024)))
        self.max_reduce_bytes = min(
            self.max_bytes, int(os.environ.get("RDMA_AR_MAX_REDUCE_BYTES", str(512 * 1024)))
        )
        self.twoshot_min_bytes = int(os.environ.get("RDMA_AR_TWOSHOT_MIN_BYTES", str(32 * 1024)))
        gid = int(os.environ.get("RDMA_AR_GID_INDEX", os.environ.get("NCCL_IB_GID_INDEX", "5")))
        core = int(os.environ.get("RDMA_AR_PROXY_CORE", "-1"))
        timeout_ms = int(os.environ.get("RDMA_AR_TIMEOUT_MS", "60000"))
        dev = device.index if device.index is not None else torch.cuda.current_device()
        self.world = world
        self.h = ext.RdmaAR(
            rank, world, self.max_bytes, self.max_reduce_bytes, hcas, gid, 1, core, timeout_ms, dev
        )
        infos = [None] * world
        dist.all_gather_object(infos, self.h.local_info(), group=cpu_group)
        self.h.connect(infos)
        dist.barrier(group=cpu_group)

    def should_use(self, x: torch.Tensor) -> bool:
        nbytes = x.numel() * x.element_size()
        return (
            x.dtype == torch.bfloat16
            and x.is_cuda
            and x.is_contiguous()
            and 0 < nbytes <= self.max_reduce_bytes
            and nbytes % 16 == 0
        )

    def all_reduce(self, x: torch.Tensor, two_shot=None) -> torch.Tensor:
        if two_shot is None:
            two_shot = 0 < self.twoshot_min_bytes <= x.numel() * x.element_size()
        return self.h.all_reduce(x, two_shot)

    def should_gather(self, out: torch.Tensor, x: torch.Tensor) -> bool:
        nbytes = x.numel() * x.element_size()
        return (
            x.is_cuda
            and x.is_contiguous()
            and out.is_contiguous()
            and out.dtype == x.dtype
            and out.numel() == x.numel() * self.world
            and 0 < nbytes <= self.max_bytes
            and nbytes % 16 == 0
        )

    def all_gather_into(self, out: torch.Tensor, x: torch.Tensor) -> None:
        self.h.all_gather_into(out, x)
