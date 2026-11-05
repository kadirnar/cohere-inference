"""The CUDA kernels in kernels/csrc, compiled on first import for this GPU (torch's JIT extension builder, cached in
~/.cache/torch_extensions; about a minute the first time)."""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SOURCES = ["bindings.cpp", "encoder_loop.cpp", "gemm.cu", "gemm_int8.cu", "encoder_ops.cu", "decoder_ops.cu", "dec_w8.cu",
            "features.cu", "gemm_sparse.cu"]


def _build():
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return load(name=f"cohere_cuda_sm{arch}", sources=[os.path.join(_DIR, s) for s in _SOURCES],
                extra_cflags=["-O3"], extra_include_paths=[_DIR],
                extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                                   "--expt-relaxed-constexpr", "-std=c++17"],
                verbose=os.environ.get("CUDA_EXT_VERBOSE") == "1")


ext = _build()

EPI_BIAS, EPI_SILU, EPI_RELU, EPI_RESID, EPI_GLU, EPI_SILU_Q8, EPI_CONV_RELU = range(7)
