@echo off
rem GPU spike build: full vcvars env + nvcc 3-arch fatbin (sm_86/89/120).
rem Output: build/gpu-spike/gpu_spike.exe (static cudart, needs opencv_world4100.dll at runtime)
setlocal
cd /d "%~dp0..\.."
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [build] vcvars64 failed
    exit /b 1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\nvcc.exe"
if not exist "%NVCC%" (
    echo [build] nvcc missing: %NVCC%
    exit /b 1
)
if not exist "build\gpu-spike" mkdir "build\gpu-spike"
set "SRC=benchmarks\gpu-spike\src"
set "OUT=build\gpu-spike\gpu_spike.exe"
echo [build] compiling with nvcc, 3 gencode archs, 1-2 min...
"%NVCC%" -O2 -cudart static -Xcompiler "/utf-8 /MD /EHsc" ^
    -gencode arch=compute_86,code=sm_86 ^
    -gencode arch=compute_89,code=sm_89 ^
    -gencode arch=compute_120,code=sm_120 ^
    -I"C:\opencv\build\include" ^
    "%SRC%\spike_main.cpp" "%SRC%\cpu_chain.cpp" "%SRC%\model_source.cpp" "%SRC%\gpu_chain.cu" ^
    -lopencv_world4100 -L"C:\opencv\build\x64\vc16\lib" ^
    -o "%OUT%"
if errorlevel 1 (
    echo [build] compile failed
    exit /b 1
)
echo [build] output: %CD%\build\gpu-spike\gpu_spike.exe
exit /b 0
