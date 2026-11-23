[CmdletBinding()]
param(
    [string]$OpenCvRoot = 'C:\opencv-5.0.0\opencv\build\x64\vc16',
    [string]$CudaToolkitRoot = '',
    [string]$VcVarsPath = '',
    [string]$CMakePath = 'cmake.exe',
    [string]$NinjaPath = 'ninja.exe',
    [ValidateSet('on', 'off', 'both')]
    [string]$Mode = 'both',
    # 构建目录名（build\ 下；OFF 目录后缀 -off）。W1 默认 gpuw1，W2A 用 gpuw2a
    [string]$BuildName = 'gpuw1'
)
# GPU 双模式构建（规划 .artifacts/proposals/gpu-port-phase0.md；-BuildName 选目录）：
#   ON  -> build/<name>      （VF_SHAPE_CUDA=ON，三代 fatbin + 静态 cudart + GPU 测试）
#   OFF -> build/<name>-off  （VF_SHAPE_CUDA=OFF，纯 CPU 对照，产物与现状等价）
# 不装配 dist/runtime（独立构建目录，不碰根装配产物）。
#
# 构建坑表（源出 benchmarks/gpu-spike README"环境坑"，在此固化）：
#   1) vcvars 完整环境：nvcc 驱动 cl/link/inc/lib 全套，本脚本先导入 vcvars64
#      再探测/调用 cmake-ninja（下方 environment import 段）。
#   2) -Xcompiler "/utf-8"：宿主 MSVC 源字符集须经 -Xcompiler 传递，已在
#      CMakeLists（COMPILE_LANGUAGE:CUDA 条件项）固化；.cu 内字符串字面量仍须
#      ASCII（nvcc 前端按系统代码页 GBK 词法分析，中文仅注释）。
#   3) std::printf：CUDA 翻译单元的宿主代码输出用完全限定 std::printf/
#      std::fprintf，避免与 device printf 符号混淆。
#   4) -L 而非 -LIBPATH:：裸 nvcc 直链 OpenCV 时 -LIBPATH: 不被转发（本脚本走
#      CMake/link.exe 不受影响；若直链，用 -L"C:\opencv\build\x64\vc16\lib"）。
#   5) 批处理纯 ASCII：cmd 按 GBK 误析 UTF-8 中文批处理会断行；.bat 保持纯
#      ASCII（本脚本为 PowerShell 不受限，新批处理须遵守）。
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $CudaToolkitRoot) { $CudaToolkitRoot = $env:CUDA_PATH }
if (-not $CudaToolkitRoot) { throw 'CUDA toolkit root not found: set -CudaToolkitRoot or CUDA_PATH' }
$nvcc = Join-Path $CudaToolkitRoot 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) { throw "nvcc missing: $nvcc" }
if (-not $VcVarsPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { throw "未找到 vswhere：$vswhere" }
    $install = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -utf8 | Select-Object -First 1)
    if (-not $install) { throw '未发现 MSVC C++ 工具链（VC.Tools.x86.x64）' }
    $VcVarsPath = Join-Path $install 'VC/Auxiliary/Build/vcvars64.bat'
}
foreach ($path in @($VcVarsPath, "$OpenCvRoot/lib/OpenCVConfig.cmake")) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "缺少构建依赖：$path" }
}
$cmake = (Get-Command $CMakePath -ErrorAction Stop).Source
$ninja = (Get-Command $NinjaPath -ErrorAction Stop).Source
if ($VcVarsPath -match '["&|<>\r\n]') { throw 'MSVC 环境脚本路径非法' }
# vcvars64.bat 内部调 vswhere：前置 Installer 目录，消除 PATH 未含它的报错噪音
$env:PATH = "$(Split-Path -Parent $vswhere);$env:PATH"
# 坑表 1：vcvars 完整环境导入（nvcc 的宿主编译链与 INCLUDE/LIB 全量就位）
$environmentLines = & cmd.exe /d /s /c "`"$VcVarsPath`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC 环境初始化失败' }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$compiler = (Get-Command cl.exe -ErrorAction Stop).Source
$linker = $compiler -replace 'cl\.exe$', 'link.exe'
$opencvBin = Join-Path $OpenCvRoot 'bin'

function Invoke-Build([string]$buildDir, [string]$cudaOption) {
    & $cmake -S $repoRoot -B $buildDir -G Ninja '-DCMAKE_BUILD_TYPE=Release' `
        "-DCMAKE_CXX_COMPILER=$compiler" "-DCMAKE_LINKER=$linker" "-DCMAKE_MAKE_PROGRAM=$ninja" `
        "-DCMAKE_CUDA_COMPILER=$nvcc" "-DOpenCV_DIR=$OpenCvRoot/lib" `
        "-DVF_SHAPE_CUDA=$cudaOption" '-DVF_SHAPE_BUILD_TESTS=ON'
    if ($LASTEXITCODE -ne 0) { throw "CMake 配置失败（CUDA=$cudaOption）：$buildDir" }
    & $cmake --build $buildDir --parallel
    if ($LASTEXITCODE -ne 0) { throw "构建失败（CUDA=$cudaOption）：$buildDir" }
    # 弹窗红线：exe/ctest 前 PATH 前置 OpenCV bin，防 loader 找不到 DLL 弹系统错误框
    $env:PATH = "$opencvBin;$env:PATH"
    & $cmake --build $buildDir --target test
    if ($LASTEXITCODE -ne 0) { throw "ctest 失败（CUDA=$cudaOption）：$buildDir" }
    Write-Host "模式 CUDA=$cudaOption 构建与测试通过：$buildDir"
}

if ($Mode -in @('on', 'both')) {
    $onDir = Join-Path $repoRoot "build\$BuildName"
    Invoke-Build $onDir 'ON'
    # 三代 fatbin 编译门（phase-0 §W5 本地化）：ON 构建后 cuobjdump 断言
    # sm_86/89/120 三代 SASS 齐，缺席即失败（批次/发版门经本脚本自然覆盖）
    & (Join-Path $repoRoot 'scripts\verify_fatbin.ps1') -Target (Join-Path $onDir 'visionforge_shape.dll') -CudaToolkitRoot $CudaToolkitRoot
}
if ($Mode -in @('off', 'both')) {
    Invoke-Build (Join-Path $repoRoot "build\$BuildName-off") 'OFF'
}
