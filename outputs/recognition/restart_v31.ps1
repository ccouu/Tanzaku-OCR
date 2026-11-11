$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$statusPath = Join-Path $workspace 'outputs\recognition\v31_pipeline_status.json'
function Write-State($state,$detail) {
    @{state=$state;detail=$detail;updated_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}
try {
    $smoke = Get-Content -LiteralPath 'outputs/recognition/pilots/v31_routing_smoke/status.json' -Raw | ConvertFrom-Json
    if ($smoke.state -ne 'completed') { throw 'Corrected routing smoke is not complete.' }
    if (Test-Path -LiteralPath 'outputs/recognition/pilots/v31_curated_short_2000') { throw 'Refusing to overwrite an existing pilot.' }
    Write-State 'training' 'Corrected routing smoke passed; preparing and running two2000-update arms.'
    & $python 'outputs/recognition/curated_short_pilot.py'
    if ($LASTEXITCODE -ne 0) { throw 'Corrected pilot failed; inspect pilot status and restart stderr.' }
    Write-State 'completed' 'Both2000-update arms complete; pilot report.json and summary.md available.'
} catch {
    Write-State 'failed' $_.Exception.Message
    throw
}
