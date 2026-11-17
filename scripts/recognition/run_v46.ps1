$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
$phase = 'training'
try {
    & $python '-m' 'scripts.recognition.train_v46_focused' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V46 training/evaluation failed; cleanup was not run.' }
    $phase = 'post-training cleanup'
    & $python '-m' 'scripts.recognition.cleanup_after_v46' '--execute'
    if ($LASTEXITCODE -ne 0) { throw 'V46 completed; conservative cleanup failed or was blocked; inspect cleanup logs.' }
    $notification = 'V46 focused recurrent training and post-training cleanup completed. Review outputs/recognition/training/v46_focused_recurrent_4000/analysis.md,report.json,sampling.json,initial_parity.json,metrics.jsonl and outputs/recognition/cleanup/v46_post_training/report.json,deletion_manifest.json. Verify actual updates/time limit versus4000maximum,30Latin/10parentheses/15digits/5basicJapanese/20YOLO/20synthetic planned mix,actual term/font/crop coverage,200newrow-onlywarmup,frozenCNN/projection/temporal/BNaffine/147buffers,trainableBiLSTM+classifier andfullsoftmaxV43teacherKL. Compare raw/EMAselected/final with promotedV43; report retention and Latin/paren/digit/basic/YOLO performance. V43remains promoted,noautopromotion,nofurthertraining. Cleanup removed only audited disposable data; verify bytesreclaimed and preserved sources/models. One-shotnotification;no polling.'
} catch {
    $notification = "V46 task encountered a failure during $phase. Inspect outputs/recognition/v46.stderr.log,training/v46_focused_recurrent_4000/status.json and cleanup/v46_post_training/. Resume only authorized focused run safely; no promotion or additional architecture unfreezing. PreserveV43/V38/allsourcecheckpoints. Cleanup only after confirmed training and evaluation completion. One-shotnotification;no polling."
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;phase=$phase;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v46_notification.json' -Encoding utf8
    if ($notification) {
        & herdr agent prompt 'w2:p1' $notification
        if ($LASTEXITCODE -ne 0) {
            @{delivered=$false;reason='Herdr client/server permission or delivery failure; local completion message preserved.';time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v46_notification_delivery.json' -Encoding utf8
        }
    }
}
