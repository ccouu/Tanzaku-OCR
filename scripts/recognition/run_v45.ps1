$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
try {
    & $python '-m' 'scripts.recognition.train_v45_latin_head' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V45 failed; inspect outputs/recognition/v45.stderr.log and pilot status.json.' }
    $notification = 'V45 Stage1A Latin head-only pilot completed. Review outputs/recognition/pilots/v45_latin_head_only_1000/analysis.md,report.json,sampling.json,initial_parity.json,metrics.jsonl and step500/1000 evaluations. Verify only33appendedLatinrows trained from V43finalraw,allpretrainedweights/BNaffine/147buffers bit-identical; 40Latin/40synthetic/20YOLO exact per500updates; blank-minus8,LR1e-3,100warmup; fullsoftmaxKLweight1validold-onlyreplayincludingYOLO. Compare new33 versus existing19Latin,fit208/train/heldout,CER/D-I-S/repeats/falseemissions and17retentiongatesbothprecisions. V43alreadyfailsabsolutegates; distinguishsource-relativeselectionfromabsolutepass. NoStage1B,nopromotion,nofurthertraining. One-shotnotification;no polling.'
} catch {
    $notification = 'V45 Stage1A Latin head-only pilot failed. Inspect outputs/recognition/v45.stderr.log and pilots/v45_latin_head_only_1000/status.json;fix safely and resume the authorized1000update maximum task. Preserve V43/V38/registry;only33newLatinrowsmaytrain. NoStage1Bauthorized. One-shotnotification;no polling.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v45_notification.json' -Encoding utf8
    if ($notification) { & herdr agent prompt 'w2:p1' $notification }
}
