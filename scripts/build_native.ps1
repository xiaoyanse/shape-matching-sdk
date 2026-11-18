[CmdletBinding()]
param(
    [string]$OpenCvRoot = 'C:\opencv\build\x64\vc16',
    [string]$VcVarsPath = '',
    [string]$CMakePath = 'cmake.exe',
    [string]$NinjaPath = 'ninja.exe',
    [string]$BuildDirectory = '',
    [switch]$Avx2,
    [switch]$BuildTests,
    [switch]$ProfileStages,
    [switch]$SkipAssemble
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDir = $repoRoot
$buildDir = Join-Path $repoRoot 'build'
if ($BuildDirectory) { $buildDir = [System.IO.Path]::GetFullPath($BuildDirectory) }
if ($Avx2 -and -not $SkipAssemble) {
    throw 'AVX2-only 实验构建须指定 -SkipAssemble，不能替换通用运行库'
}
if ($ProfileStages -and -not $SkipAssemble) {
    throw '诊断计时构建须指定 -SkipAssemble，不能替换运行库'
}
$runtimeDir = Join-Path $repoRoot 'dist/runtime'
if (-not $VcVarsPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { throw "未找到 vswhere：$vswhere" }
    $install = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -utf8 | Select-Object -First 1)
    if (-not $install) { throw '未发现 MSVC C++ 工具链（VC.Tools.x86.x64）' }
    $VcVarsPath = Join-Path $install 'VC/Auxiliary/Build/vcvars64.bat'
}
$vendorManifest = Get-Content -LiteralPath "$sourceDir/vendor-integrity.json" -Raw | ConvertFrom-Json
foreach ($entry in $vendorManifest.files) {
    if ($entry.path -match '(^/|\\|\.\.)') { throw '上游清单路径非法' }
    $actual = (Get-FileHash -LiteralPath "$sourceDir/vendor/shape_based_matching/$($entry.path)" -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.sha256) { throw "上游源码字节不匹配：$($entry.path)" }
}
# sdk_version 单一来源 = CMake project() VERSION；解析失败即中止，不备第二份硬编码。
$cmakeSource = Get-Content -LiteralPath (Join-Path $sourceDir 'CMakeLists.txt') -Raw
if ($cmakeSource -notmatch '(?m)^project\s*\([^)]*?VERSION\s+(\d+\.\d+\.\d+)[^)]*\)') {
    throw 'CMakeLists.txt 缺少 project(... VERSION x.y.z)，无法确定 sdk_version'
}
$sdkVersion = $Matches[1]
foreach ($path in @($VcVarsPath, "$OpenCvRoot/lib/OpenCVConfig.cmake")) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "缺少构建依赖：$path" }
}
$cmake = (Get-Command $CMakePath -ErrorAction Stop).Source
$ninja = (Get-Command $NinjaPath -ErrorAction Stop).Source
if ($VcVarsPath -match '["&|<>\r\n]') { throw 'MSVC 环境脚本路径非法' }
$environmentLines = & cmd.exe /d /s /c "`"$VcVarsPath`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC 环境初始化失败' }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$compiler = (Get-Command cl.exe -ErrorAction Stop).Source
# 显式指定 linker：避免残留 VS 目录的幽灵 link.exe 误导 CMake 的 vs_link_exe，
# 固定用编译器同目录 link.exe，清缓存重配也不漂移。
$linker = $compiler -replace 'cl\.exe$', 'link.exe'
$msvcToolsVersion = $env:VCToolsVersion
$cmakeVersion = ((& $cmake --version) | Select-Object -First 1) -replace '^cmake version\s+', ''
$ninjaVersion = (& $ninja --version | Select-Object -First 1)
$avx2Option = if ($Avx2) { 'ON' } else { 'OFF' }
$testOption = if ($BuildTests) { 'ON' } else { 'OFF' }
$profileOption = if ($ProfileStages) { 'ON' } else { 'OFF' }
& $cmake -S $sourceDir -B $buildDir -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_CXX_COMPILER=$compiler" "-DCMAKE_LINKER=$linker" "-DCMAKE_MAKE_PROGRAM=$ninja" "-DOpenCV_DIR=$OpenCvRoot/lib" "-DVF_SHAPE_AVX2=$avx2Option" "-DVF_SHAPE_BUILD_TESTS=$testOption" "-DVF_SHAPE_PROFILE_STAGES=$profileOption"
if ($LASTEXITCODE -ne 0) { throw '原生 CMake 配置失败' }
& $cmake --build $buildDir --clean-first --parallel
if ($LASTEXITCODE -ne 0) { throw '原生匹配器编译失败' }
if ($BuildTests) {
    # 测试 exe 按 PATH 解析 opencv_world 运行库，注入 OpenCV bin 防止
    # 0xc0000135（找不到 DLL）并触发系统弹窗。
    $env:PATH = "$(Join-Path $OpenCvRoot 'bin');$env:PATH"
    & $cmake --build $buildDir --target test
    if ($LASTEXITCODE -ne 0) { throw '原生测试失败' }
}
if ($SkipAssemble) {
    Write-Host "原生组件已构建，未装配：$buildDir"
    return
}
New-Item -ItemType Directory -Force $runtimeDir | Out-Null
Copy-Item -LiteralPath (Join-Path $buildDir 'visionforge_shape.dll') -Destination $runtimeDir -Force
$opencvLibraries = @(Get-ChildItem -LiteralPath "$OpenCvRoot/bin" -Filter 'opencv_world*.dll' |
    Where-Object { $_.Name -notmatch 'd\.dll$' })
if ($opencvLibraries.Count -ne 1) { throw '需要唯一 Release opencv_world DLL' }
Copy-Item -LiteralPath $opencvLibraries[0].FullName -Destination $runtimeDir -Force
$opencvVersion = ''
if ($opencvLibraries[0].Name -match '^opencv_world(\d+)\.dll$') {
    $digits = $Matches[1]
    if ($digits.Length -ge 4) {
        $opencvVersion = '{0}.{1}.{2}' -f $digits.Substring(0, 1), $digits.Substring(1, 2), $digits.Substring(3)
    } elseif ($digits.Length -eq 3) {
        $opencvVersion = '{0}.{1}.{2}' -f $digits.Substring(0, 1), $digits.Substring(1, 1), $digits.Substring(2)
    }
}
if (-not $opencvVersion) { $opencvVersion = $opencvLibraries[0].Name }
Copy-Item -LiteralPath "$sourceDir/vendor/shape_based_matching/LICENSE" -Destination "$runtimeDir/shape_based_matching.LICENSE" -Force
Copy-Item -LiteralPath "$sourceDir/THIRD_PARTY_NOTICES.md" -Destination $runtimeDir -Force
$opencvLicense = Join-Path $OpenCvRoot '../../LICENSE'
if (-not (Test-Path -LiteralPath $opencvLicense)) { throw '缺少 OpenCV 许可证' }
Copy-Item -LiteralPath $opencvLicense -Destination "$runtimeDir/OpenCV.LICENSE" -Force
$files = [ordered]@{}
foreach ($path in @('visionforge_shape.dll', $opencvLibraries[0].Name,
    'shape_based_matching.LICENSE', 'THIRD_PARTY_NOTICES.md', 'OpenCV.LICENSE')) {
    $files[$path] = (Get-FileHash -LiteralPath (Join-Path $runtimeDir $path) -Algorithm SHA256).Hash.ToLowerInvariant()
}
# 版本锚点：构建时 git 提交号与最近 tag，供消费方自动检测上游更新。
$revision = ''
$anchor = ''
$worktreeDirty = $false
if (Test-Path (Join-Path $repoRoot '.git')) {
    $revision = (& git -C $repoRoot rev-parse --short HEAD 2>$null)
    if ($LASTEXITCODE -ne 0) { $revision = '' }
    $anchor = (& git -C $repoRoot describe --tags --always 2>$null)
    if ($LASTEXITCODE -ne 0) { $anchor = '' }
    $worktreeEntries = @(& git -C $repoRoot status --porcelain)
    $worktreeDirty = ($LASTEXITCODE -eq 0 -and $worktreeEntries.Count -gt 0)
}
$manifest = [ordered]@{
    abi_version    = 3
    sdk_version    = $sdkVersion
    platform       = 'windows-x64'
    sdk_revision   = $revision
    anchor         = $anchor
    worktree_dirty = $worktreeDirty
    toolchain      = [ordered]@{
        msvc_tools = $msvcToolsVersion
        cmake      = $cmakeVersion
        ninja      = $ninjaVersion
        opencv     = $opencvVersion
    }
    files          = $files
}
# ConvertTo-Json 多行输出为 CRLF，统一 LF，避免消费方哈希链被行尾改写打断。
$json = ($manifest | ConvertTo-Json -Depth 4) -replace "`r`n", "`n"
[System.IO.File]::WriteAllText((Join-Path $runtimeDir 'runtime-manifest.json'), $json + "`n", [System.Text.UTF8Encoding]::new($false))
if ($BuildTests) {
    # Python 主链门禁以本批刚装配的 dist/runtime 为唯一测试对象，显式指定，
    # 覆盖调用方环境残留的 SHAPE_SDK_RUNTIME；-SkipAssemble 时不具备被测对象，不进入本分支。
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) { throw 'Python 主链门禁需要 python（含 numpy），未在 PATH 找到 python' }
    & $python.Source -c "import numpy; assert int(numpy.__version__.split('.')[0]) == 2, 'numpy ' + numpy.__version__ + ' 不符绑定契约 numpy>=2,<3'"
    if ($LASTEXITCODE -ne 0) { throw 'Python 主链门禁依赖 numpy>=2,<3 未满足（缺失或主版本不符），环境未按契约准备' }
    $env:SHAPE_SDK_RUNTIME = $runtimeDir
    & $python.Source -B (Join-Path $sourceDir 'scripts/test_main_chain.py')
    if ($LASTEXITCODE -ne 0) { throw 'Python 主链回归失败' }
}
Write-Host "原生组件已构建并装配：$runtimeDir"
