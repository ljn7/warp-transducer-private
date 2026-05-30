# install.ps1 - Build warp-transducer and install the PyTorch binding on Windows.
#
# Usage (from PowerShell in the repo root):
#   .\install.ps1               # auto-detect CUDA, build everything, pip install
#   .\install.ps1 -NoCuda       # force CPU-only build
#   .\install.ps1 -NoInstall    # build C library only, skip pip install
#   .\install.ps1 -BuildDir D:\build  # custom build directory
#
# Requirements:
#   cmake >= 3.10 (in PATH or installed via Visual Studio)
#   Visual Studio 2019/2022 Build Tools with "Desktop development with C++"
#   Python with PyTorch installed  (conda or venv both work)
#   Optional: CUDA toolkit (detected automatically)

[CmdletBinding()]
param(
    [switch]$NoCuda,
    [switch]$NoInstall,
    [string]$BuildDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $BuildDir) {
    $BuildDir = Join-Path $ScriptDir "build"
}

$BindingDir = Join-Path $ScriptDir "pytorch_binding"

Write-Host ""
Write-Host "============================================================"
Write-Host " warp-transducer Windows install"
Write-Host " PowerShell: $($PSVersionTable.PSVersion)"
Write-Host " Build dir:  $BuildDir"
Write-Host "============================================================"

# ---------------------------------------------------------------------------
# Helper: die with a clear message
# ---------------------------------------------------------------------------
function Fail($msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    Write-Host ""
    exit 1
}

# ---------------------------------------------------------------------------
# Helper: run a command and stop on failure
# ---------------------------------------------------------------------------
function Run($cmd, $cwd = $null, $desc = "") {
    $label = if ($desc) { $desc } else { $cmd }
    Write-Host ""
    Write-Host ">>> $label" -ForegroundColor Cyan
    if ($cwd) {
        Push-Location $cwd
    }
    try {
        Invoke-Expression $cmd
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            Fail "Command failed (exit $LASTEXITCODE): $label"
        }
    }
    finally {
        if ($cwd) { Pop-Location }
    }
}

# ---------------------------------------------------------------------------
# Step 0: Check cmake
# ---------------------------------------------------------------------------
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    # Try common VS install paths
    $vsCmakePaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )
    foreach ($p in $vsCmakePaths) {
        if (Test-Path $p) {
            $cmake = $p
            $env:PATH = (Split-Path $p) + ";" + $env:PATH
            break
        }
    }
}
if (-not $cmake) {
    Fail @"
cmake not found in PATH.

Install it one of these ways:
  1. Download from https://cmake.org/download/ and add to PATH, OR
  2. Install Visual Studio with the 'C++ CMake tools for Windows' component
     (Visual Studio Installer -> Modify -> Individual Components -> 'C++ CMake tools')
"@
}
$cmakeVersion = (& cmake --version 2>&1 | Select-Object -First 1).ToString().Split()[2]
Write-Host " CMake:    $cmakeVersion" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 0b: Find Ninja (preferred generator on Windows - avoids needing vcvars)
# ---------------------------------------------------------------------------
$ninja = Get-Command ninja -ErrorAction SilentlyContinue
if (-not $ninja) {
    $ninjaSearchPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
    )
    foreach ($p in $ninjaSearchPaths) {
        $candidate = Join-Path $p "ninja.exe"
        if (Test-Path $candidate) {
            $ninja = $candidate
            $env:PATH = $p + ";" + $env:PATH
            break
        }
    }
}
if ($ninja) {
    Write-Host " Ninja:    found ($ninja)" -ForegroundColor Green
    $Generator = "Ninja"
} else {
    Write-Host " Ninja:    not found, falling back to NMake Makefiles" -ForegroundColor Yellow
    $Generator = "NMake Makefiles"
}

# ---------------------------------------------------------------------------
# Step 0c: Find cl.exe (MSVC compiler)
# ---------------------------------------------------------------------------
$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $cl) {
    Write-Host ""
    Write-Host "WARNING: cl.exe (MSVC compiler) not found in PATH." -ForegroundColor Yellow
    Write-Host "         cmake might still find it via vswhere."
    Write-Host "         If the build fails, run from a 'Developer PowerShell for VS 2022' or"
    Write-Host "         'x64 Native Tools Command Prompt for VS 2022'."
} else {
    $clVersion = (& cl /? 2>&1 | Select-Object -First 2 | Select-Object -Last 1)
    Write-Host " MSVC:     $clVersion" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Step 0d: Detect CUDA
# ---------------------------------------------------------------------------
$WithGpu = "OFF"
$CudaRoot = $null

if (-not $NoCuda) {
    # Check environment variables set by CUDA installer
    foreach ($var in @("CUDA_PATH", "CUDA_HOME")) {
        $val = [Environment]::GetEnvironmentVariable($var)
        if ($val -and (Test-Path $val)) {
            $CudaRoot = $val
            break
        }
    }

    # Check the standard CUDA install location
    if (-not $CudaRoot) {
        $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $cudaBase) {
            $versions = Get-ChildItem $cudaBase -Directory | Sort-Object Name -Descending
            if ($versions) {
                $CudaRoot = $versions[0].FullName
            }
        }
    }

    # Check if nvcc is in PATH
    if (-not $CudaRoot) {
        $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
        if ($nvcc) {
            # nvcc is at <cuda_root>\bin\nvcc.exe
            $CudaRoot = Split-Path (Split-Path $nvcc.Source)
        }
    }

    if ($CudaRoot) {
        $nvccPath = Join-Path $CudaRoot "bin\nvcc.exe"
        if (Test-Path $nvccPath) {
            $cudaVersion = (& $nvccPath --version 2>&1 | Select-String "release").ToString().Trim()
            Write-Host " CUDA:     $cudaVersion" -ForegroundColor Green
            Write-Host "           Root: $CudaRoot" -ForegroundColor Green
            $WithGpu = "ON"
            # Ensure nvcc is in PATH for cmake CUDA detection
            $env:PATH = (Join-Path $CudaRoot "bin") + ";" + $env:PATH
            $env:CUDA_PATH = $CudaRoot
        } else {
            Write-Host " CUDA:     directory found but nvcc missing - treating as not found" -ForegroundColor Yellow
            $CudaRoot = $null
        }
    } else {
        Write-Host " CUDA:     not detected (will build CPU-only)" -ForegroundColor Yellow
        Write-Host "           To enable GPU: install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
    }
} else {
    Write-Host " GPU:      disabled by -NoCuda flag"
}

Write-Host "============================================================"

# ---------------------------------------------------------------------------
# Step 1: cmake configure
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$cmakeArgs = @(
    $ScriptDir,
    "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DWITH_GPU=$WithGpu"
)

Run ("cmake " + ($cmakeArgs -join " ")) $BuildDir "CMake configure"

# ---------------------------------------------------------------------------
# Step 2: cmake build
# ---------------------------------------------------------------------------
Run "cmake --build . --config Release" $BuildDir "Build C library"

# ---------------------------------------------------------------------------
# Step 3: verify the library exists
# ---------------------------------------------------------------------------
$DllPath = Join-Path $BuildDir "warprnnt.dll"
if (-not (Test-Path $DllPath)) {
    Fail @"
Build finished but warprnnt.dll was not found in:
  $BuildDir

Check the cmake/build output above for errors.
"@
}
Write-Host ""
Write-Host "C library built: $DllPath" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 4: pip install the PyTorch binding
# ---------------------------------------------------------------------------
if ($NoInstall) {
    Write-Host ""
    Write-Host "Skipping pip install (-NoInstall was set)."
    Write-Host "To install later, run from pytorch_binding/:"
    Write-Host "  `$env:WARP_RNNT_PATH='$BuildDir'; pip install ."
    exit 0
}

# Check Python / pip
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Fail "python not found in PATH. Make sure Python (with PyTorch) is activated."
}

$torchCheck = & python -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Fail @"
PyTorch is not installed in the current Python environment.
Install it from https://pytorch.org/get-started/locally/ then re-run this script.
"@
}
Write-Host " Python:   $(& python --version 2>&1)" -ForegroundColor Green
Write-Host " PyTorch:  $torchCheck" -ForegroundColor Green

$env:WARP_RNNT_PATH = $BuildDir
if ($CudaRoot) {
    $env:CUDA_PATH = $CudaRoot
    $env:CUDA_HOME = $CudaRoot
}

Run "python -m pip install ." $BindingDir "pip install pytorch binding"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================"
Write-Host " Installation complete." -ForegroundColor Green
Write-Host ""
Write-Host " Verify it works:"
Write-Host "   python -c `"from warprnnt_pytorch import RNNTLoss; print('OK')`""
Write-Host ""
Write-Host " Quick usage example:"
Write-Host "   import torch"
Write-Host "   from warprnnt_pytorch import RNNTLoss"
Write-Host "   loss_fn = RNNTLoss()"
Write-Host "   print('warp-transducer ready')"
Write-Host "============================================================"
