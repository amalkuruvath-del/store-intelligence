<#
.SYNOPSIS
Batch runner for the Store Intelligence detection pipeline on Windows.

.DESCRIPTION
Discovers all video clips in the specified data directory and runs pipeline\detect.py on each.

.PARAMETER DataDir
Directory containing .mp4 / .avi video clips (default: data)

.PARAMETER StoreId
Store identifier (default: STORE_BLR_001)

.PARAMETER LayoutPath
Path to store_layout.json (default: data\store_layout.json)
#>

param(
    [string]$DataDir = "data",
    [string]$StoreId = "STORE_BLR_001",
    [string]$LayoutPath = "",
    [string]$ForceDate = ""
)

if ($LayoutPath -eq "") {
    $LayoutPath = Join-Path $DataDir "store_layout.json"
}

$ErrorActionPreference = "Stop"

# Colors
function Write-Info($msg) { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-ErrorMsg($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# API check
$ApiUrl = $env:SIS_API_URL
if (-not $ApiUrl) { $ApiUrl = "http://localhost:8000" }

Write-Info "Checking API health at $ApiUrl/health ..."
try {
    $response = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 5
    Write-Info "API is reachable."
} catch {
    Write-Warn "API at $ApiUrl is NOT reachable."
    Write-Warn "Events will be emitted but may fail to POST."
    Write-Warn "Continuing anyway - events are logged locally."
}

# Discover clips
if (-not (Test-Path -Path $DataDir)) {
    Write-ErrorMsg "Data directory not found: $DataDir"
    exit 1
}

$clips = Get-ChildItem -Path $DataDir -Include *.mp4,*.avi,*.mkv,*.mov -Recurse | Where-Object { -not $_.PSIsContainer }

if ($clips.Count -eq 0) {
    Write-ErrorMsg "No video clips found in $DataDir"
    exit 1
}

Write-Info "Found $($clips.Count) clip(s) in $DataDir"

$processed = 0
$failed = 0

# ── 3. Process clips using the Python Batch Runner ─────────────
Write-Info "Delegating batch execution to Python so Re-ID memory is shared..."
$proc = Start-Process -FilePath "python" -ArgumentList "-m pipeline.batch_run --data-dir `"$DataDir`" --store-id `"$StoreId`" --layout `"$LayoutPath`"" -NoNewWindow -Wait -PassThru

exit $proc.ExitCode
