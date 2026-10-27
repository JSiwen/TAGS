#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
#
# TAGSs patches (2026-10-16):
# 1. Add `-allow-unsupported-compiler` to nvcc flags so nvcc 11.8 accepts
#    MSVC 2022 14.40+ (without this, host_config.h rejects with C1189).
# 2. Add `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1` define so MSVC STL
#    14.40+ static-assertion STL1002 (which requires CUDA 12.4+) does not
#    fire. Combined with (1), this lets nvcc 11.8 + MSVC 14.44 build
#    simple_knn on Python 3.7.1 / torch 1.12.1+cu116 / Windows.
# 3. Note: distCUDA2 only runs once at GaussianModel init; densify hot path
#    (densify_and_clone / densify_and_split) is pure GPU and unaffected.
#    If MSVC/nvcc toolchain regresses, `TAGGS_CPU_KNN_INIT=1` in PS1
#    wrappers falls back to scipy cKDTree at ~1.6 min warmup cost.
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

# nvcc extra args: 兼容更新的 MSVC (2022 14.40+), nvcc 11.8 host_config.h 默认拒绝
_nvcc_flags = []
if os.name == 'nt':
    _nvcc_flags.append("-allow-unsupported-compiler")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu",
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": _nvcc_flags, "cxx": cxx_compiler_flags},
            define_macros=[("_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", "1")])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
