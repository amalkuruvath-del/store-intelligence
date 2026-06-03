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
    [string]$LayoutPath = "data\store_layout.json",
    [string]$ForceDate = ""
)

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

foreach ($clip in $clips) {
    $filename = $clip.Name
    $name_no_ext = $clip.BaseName

    if ($name_no_ext -match "(.*?)__(.*)") {
        $CameraId = $matches[1]
    } else {
        $CameraId = $name_no_ext
    }
    
    # Clean up name, remove spaces or weird characters if needed (for safety, though API can handle it)
    $CameraId = $CameraId.Replace(" ", "_")

    if ($name_no_ext -match "([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2})") {
        $raw_ts = $matches[1]
        $StartTime = $raw_ts.Substring(0, 11) + $raw_ts.Substring(11, 2) + ":" + $raw_ts.Substring(14, 2) + ":" + $raw_ts.Substring(17, 2) + "Z"
    } else {
        $StartTime = $clip.LastWriteTimeUtc.ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
    
    if ($ForceDate -ne "") {
        $timePart = $StartTime.Substring(11)
        $StartTime = "$ForceDate`T$timePart"
    }

    Write-Info "----------------------------------------------------"
    Write-Info "Processing: $($clip.FullName)"
    Write-Info "  Store:    $StoreId"
    Write-Info "  Camera:   $CameraId"
    Write-Info "  Start:    $StartTime"
    Write-Info "  Layout:   $LayoutPath"
    Write-Info "----------------------------------------------------"

    try {
        # Run python detection
        $proc = Start-Process -FilePath "python" -ArgumentList "-m pipeline.detect --video `"$($clip.FullName)`" --store-id `"$StoreId`" --camera-id `"$CameraId`" --layout `"$LayoutPath`" --start-time `"$StartTime`"" -NoNewWindow -Wait -PassThru
        
        if ($proc.ExitCode -eq 0) {
            $processed++
        } else {
            Write-ErrorMsg "Failed to process $($clip.FullName) with exit code $($proc.ExitCode)"
            $failed++
        }
    } catch {
        Write-ErrorMsg "Failed to start python pipeline: $_"
        $failed++
    }
}

Write-Info ""
Write-Info "======================================================="
Write-Info "               PIPELINE RUN COMPLETE"
Write-Info "======================================================="
Write-Info "  Total clips:     $($clips.Count)"
Write-Info "  Processed OK:    $processed"
if ($failed -gt 0) {
    Write-ErrorMsg "  Failed:          $failed"
} else {
    Write-Info "  Failed:          0"
}
Write-Info "======================================================="

if ($failed -gt 0) {
    exit $failed
}
