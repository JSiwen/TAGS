@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 10.0.19041.0 -vcvars_ver=14.29
if errorlevel 1 exit /b 1

set "CUDA_HOME=C:\Users\yori\miniconda3\envs\gaussian_splatting"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "NVCC_FLAGS=-allow-unsupported-compiler"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\Library\bin;%PATH%"
set "PYTHON=C:\Users\yori\miniconda3\envs\gaussian_splatting\python.exe"
set "ROOT=d:\TAGSs\TAGSs_core"

"%PYTHON%" -m pip install ninja
if errorlevel 1 exit /b 1

"%PYTHON%" -m pip install "%ROOT%\submodules\diff-gaussian-rasterization"
if errorlevel 1 exit /b 1

"%PYTHON%" -m pip install "%ROOT%\submodules\simple-knn"
if errorlevel 1 exit /b 1

"%PYTHON%" -m pip install "%ROOT%\submodules\fused-ssim"
if errorlevel 1 exit /b 1

echo All CUDA submodules installed successfully.
