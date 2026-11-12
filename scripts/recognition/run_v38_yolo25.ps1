$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
try {
    & $python '-m' 'scripts.recognition.train_v38_yolo25' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V38 training failed; inspect v38.stderr.log and training status.json.' }
    $notification = 'V38 YOLO25 training completed. Review outputs/recognition/training/v38_yolo25_fullpasses4/analysis.md, report.json, sampling.json, epoch_audits.json, metrics.jsonl and initial_parity.json. Verify 686 updates / 21376 presentations, 1336 eligible YOLO train crops once in each of four passes (5344,25%),135 heldout crops from13 menu-disjoint validation pages,521 exact terms per pass and147 fixed BN buffers for raw/EMA. Compare initial V36, selected best.pt and final raw/EMA; assess train/holdout CER and retention. Labels are VLM-generated, not independently verified. No further training running. One-shot completion notification; no polling.'
} catch {
    $notification = 'V38 YOLO25 training failed. Continue authorized task by inspecting outputs/recognition/v38.stderr.log and training/v38_yolo25_fullpasses4/status.json. Fix safely and resume; preserve V36 and all original checkpoints. Authorized four complete1336-crop passes,25% YOLO mixture,686 planned updates. One-shot failure notification.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v38_notification.json' -Encoding utf8
    if ($notification) {
        & herdr agent prompt 'w2:p1' $notification
    }
}
