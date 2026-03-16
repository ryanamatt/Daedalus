# tools/clean_builds.ps1

$ErrorActionPreference = "Stop"

Write-Host "--- Starting Daedadlus Deep Clean ---" -ForegroundColor Cyan

# Remove Build Artifacts
Write-Host "Removing build directories..."
$Targets = @("build", "dist", ".pytest_cache", "daedalus.egg_info")
foreach ($Target in $Targets) {
    if (Test-Path $Target) {
        Write-Host "Removing $Target..."
        Remove-Item -Recurse -Force $Target
    }
}

# Remove Compiled Python/C++ Binaries in the source tree
Write-Host "Cleaning compiled binaries and caches..."
Get-ChildItem -Path "daedalus" -Filter "*.pyd" -Recurse | Remove-Item -Force
Get-ChildItem -Path "." -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force

# Reinstall in editable mode
Write-Host "--- Rebuilding Project ---" -ForegroundColor Cyan
pip install -e ".[test]"

Write-Host "`rBuild Complete!." -ForegroundColor Green