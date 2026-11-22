[CmdletBinding()]
param(
    # 待验目标：CUDA ON 构建的 DLL/EXE（内嵌三代 fatbin）。默认取 GPU ON 构建产物
    [string]$Target = '',
    [string]$CudaToolkitRoot = ''
)
# 三代 fatbin 编译门（phase-0 §W5 本地化，协调线裁甲 2026-11-22）：cuobjdump 断言
# 内嵌 SASS 覆盖支持集 {sm_86, sm_89, sm_120}，缺席即非零退出。挂 build_gpu.ps1
# ON 模式构建后（批次/发版门经该脚本自然覆盖）；OFF 构建无 fatbin 不适用。
# 注意：W4 起 fatbin 无 PTX 兜底（仅 -real 三代 SASS），跨更代架构走 CPU 回退——
# 支持集变更须同步 phase-0 §二探测分支与发布文案。
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $CudaToolkitRoot) { $CudaToolkitRoot = $env:CUDA_PATH }
if (-not $CudaToolkitRoot) { throw 'CUDA toolkit root not found: set -CudaToolkitRoot or CUDA_PATH' }
$cuobjdump = Join-Path $CudaToolkitRoot 'bin\cuobjdump.exe'
if (-not (Test-Path -LiteralPath $cuobjdump -PathType Leaf)) { throw "cuobjdump missing: $cuobjdump" }
if (-not $Target) {
    $candidate = Join-Path $repoRoot 'build\rel30\visionforge_shape.dll'
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "default target missing: $candidate (pass -Target explicitly)"
    }
    $Target = $candidate
}
$output = & $cuobjdump --list-elf "$Target" 2>&1
if ($LASTEXITCODE -ne 0) { throw "cuobjdump failed on $Target" }
$elfLines = @($output | Where-Object { $_ -match 'ELF file\s+\d+:.*\.sm_\d+\.cubin' })
$found = @{}
foreach ($line in $elfLines) {
    if ($line -match 'sm_(\d+)') { $found["sm_$($Matches[1])"] = $true }
}
$required = 'sm_86', 'sm_89', 'sm_120'
$missing = @($required | Where-Object { -not $found.ContainsKey($_) })
$archList = ($found.Keys | Sort-Object) -join ', '
Write-Host "fatbin SASS generations in $([System.IO.Path]::GetFileName($Target)): elf sections=$($elfLines.Count), archs=$archList"
if ($missing.Count -gt 0) {
    throw "fatbin missing required SASS: $($missing -join ', ') (found: $($found.Keys -join ', '))"
}
Write-Host "fatbin gate PASS: $($required -join '/')"
