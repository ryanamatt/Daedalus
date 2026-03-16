# tools/tree.ps1

$ExcludeList = "venv", "__pycache__", ".pytest_cache", "dist", "build", "html", ".benchmarks", ".vscode"
$DepthValue = 5

Write-Host "Running Command: PSTree -Exclude '$($ExcludeList -join ', ')' -Depth $DepthValue"

# Execute PSTree directly, passing the parameters
PSTree -Exclude $ExcludeList -Depth $DepthValue