# Windows setup for the binary-scout YOLO pipeline.
#
# Run from the repo root in PowerShell:
#   .\setup.ps1          # auto-detect CUDA
#   .\setup.ps1 cpu      # CPU-only PyTorch
#   .\setup.ps1 cu126    # CUDA 12.6 PyTorch
#
# Native kernel prerequisite:
#   Install MSYS2, open the "MSYS2 UCRT64" shell, then run:
#     pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
#   PowerShell builds need C:\msys64\ucrt64\bin on PATH.

param(
    [ValidateSet("auto", "cpu", "cu118", "cu121", "cu124", "cu126")]
    [string]$CudaBuild = "auto"
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "[setup] $Message" -ForegroundColor Cyan
}

function Write-Warn([string]$Message) {
    Write-Host "[setup] WARNING: $Message" -ForegroundColor Yellow
}

$UcrtBin = "C:\msys64\ucrt64\bin"
if (Test-Path $UcrtBin) {
    $env:PATH = "$UcrtBin;$env:PATH"
}

$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Warn "PowerShell execution policy is Restricted."
    Write-Warn "Run once: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
    exit 1
}

$pythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $pythonCmd = $cmd
        break
    }
}
if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ first." -ForegroundColor Red
    exit 1
}
Write-Step "Using $(& $pythonCmd --version 2>&1) ($pythonCmd)"

if (-not (Test-Path "venv")) {
    Write-Step "Creating virtual environment"
    & $pythonCmd -m venv venv
} else {
    Write-Step "venv already exists"
}

$activateScript = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: venv activation script not found at $activateScript" -ForegroundColor Red
    exit 1
}
. $activateScript
python -m pip install --quiet --upgrade pip

if ($CudaBuild -eq "auto") {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $CudaBuild = "cu126"
        Write-Step "nvidia-smi found; using $CudaBuild PyTorch build"
    } else {
        $CudaBuild = "cpu"
        Write-Step "No CUDA detected; using CPU-only PyTorch"
    }
}

if ($CudaBuild -eq "cpu") {
    Write-Step "Installing CPU-only PyTorch"
    pip install torch torchvision
} else {
    Write-Step "Installing PyTorch $CudaBuild"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaBuild"
}

Write-Step "Installing remaining dependencies"
$deps = Get-Content requirements.txt |
    Where-Object { $_ -notmatch '^\s*#' -and $_ -notmatch '^\s*$' -and $_ -notmatch '^torch' }
if ($deps) {
    $tmpFile = [System.IO.Path]::GetTempFileName()
    $deps | Set-Content $tmpFile
    pip install --quiet -r $tmpFile
    Remove-Item $tmpFile
}

Write-Step "Building C kernel"
$gccAvailable = Get-Command gcc -ErrorAction SilentlyContinue
$makeAvailable = Get-Command mingw32-make -ErrorAction SilentlyContinue
if ($gccAvailable -and $makeAvailable) {
    mingw32-make
    if ($LASTEXITCODE -eq 0) {
        Write-Step "kernel.dll built successfully"
    } else {
        Write-Warn "mingw32-make failed. Check MSYS2 UCRT64 gcc installation."
    }
} else {
    Write-Warn "gcc or mingw32-make not found. The C kernel will not be available."
    Write-Host "Open the MSYS2 UCRT64 shell and run:" -ForegroundColor Yellow
    Write-Host "  pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make" -ForegroundColor Yellow
    Write-Host "Then re-run setup.ps1 from PowerShell." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python scripts\download_visdrone.py --splits train val"
Write-Host "  python scripts\verify_packed_kernel.py --include-nonbinary"
Write-Host ""
Write-Host "See README.md for the complete data, feature, training, and routing commands."
