$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
try {
    & $python '-m' 'scripts.recognition.train_v41_new_rows' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V41 failed; inspect v41.stderr.log and pilot status.json.' }
    $notification = 'V41 new-row-only2000-update pilot completed. Review outputs/recognition/pilots/v41_new_rows_only_2000/analysis.md,report.json,sampling.json,new_character_coverage.json,initial_parity.json,metrics.jsonl and paired_analysis.json. Verify blank-row-minus8initialization passed zero-new-emission/zero-prediction-difference preflight in FP16/BF16; all pretrained weights,BNaffine,oldclassifierrows and147BNbuffers bit-identical through2000updates. Verify25%controlledone-new-character/25%YOLO/50%oldreplay,64000presentations,all57newrows changed and >=280newchar encounters; compare with V40 on identical684probes/156YOLOholdout and revisedretention. Teacher consistency only on valid frames of old-only labels. V38/registry unchanged,noauto-promotion. One-shot completion notification;no polling.'
} catch {
    $notification = 'V41 new-row-only pilot failed. Continue original authorized revised2000-update task by inspecting outputs/recognition/v41.stderr.log and pilots/v41_new_rows_only_2000/status.json. Fix safely and resume, preserving V38/V40 and all original checkpoints. Initialization must pass source prediction parity before any updates; no pretrained weights may train. One-shot failure notification.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v41_notification.json' -Encoding utf8
    if ($notification) { & herdr agent prompt 'w2:p1' $notification }
}
