$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$curation = Join-Path $workspace 'outputs\recognition\curation\v31_short_terms'
$statusPath = Join-Path $workspace 'outputs\recognition\v31_pipeline_status.json'
function Write-PipelineStatus($state, $detail) {
    @{ state=$state; detail=$detail; updated_utc=[DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}
try {
    Write-PipelineStatus 'awaiting_argon_review' 'No full training until review artifacts reconcile and GPU smoke test passes.'
    $deadline = [DateTime]::UtcNow.AddHours(3)
    while ($true) {
        if ([DateTime]::UtcNow -gt $deadline) { throw 'Timed out awaiting Gemini review artifacts; no training launched.' }
        $reviewPath = Join-Path $curation 'argon_review.jsonl'
        $summaryPath = Join-Path $curation 'argon_summary.md'
        if ((Test-Path -LiteralPath $reviewPath) -and (Test-Path -LiteralPath $summaryPath)) {
            try {
                $review = @(Get-Content -LiteralPath $reviewPath -Encoding utf8 | ForEach-Object { $_ | ConvertFrom-Json })
                if ($review.Count -eq 5069) { break }
            } catch { }
        }
        Start-Sleep -Seconds 10
    }
    Write-PipelineStatus 'reconciling_reviews' 'Conservative high-confidence agreement plus explicitly sourced essentials.'
    & $python 'outputs/recognition/curation/v31_short_terms/reconcile.py'
    if ($LASTEXITCODE -ne 0) { throw 'Review reconciliation failed; no full training launched.' }
    Write-PipelineStatus 'gpu_smoke' 'Reconciled-list smoke:10 updates per arm.'
    & $python 'outputs/recognition/curated_short_pilot.py' '--smoke' '--steps' '10' '--output' 'outputs/recognition/pilots/v31_reconciled_smoke'
    if ($LASTEXITCODE -ne 0) { throw 'Reconciled GPU smoke failed; no full training launched.' }
    Write-PipelineStatus 'training' 'Two sequential2000-update arms; see pilot status and metrics files.'
    & $python 'outputs/recognition/curated_short_pilot.py'
    if ($LASTEXITCODE -ne 0) { throw 'V31 pilot failed; inspect pilot status.json and stderr.' }
    Write-PipelineStatus 'completed' 'Both arms finished; report.json and summary.md available in v31_curated_short_2000.'
} catch {
    Write-PipelineStatus 'failed' $_.Exception.Message
    throw
}
