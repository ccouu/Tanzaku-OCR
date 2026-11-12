$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
try {
    & $python '-m' 'scripts.recognition.train_v42_yolo35' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V42 failed; inspect v42.stderr.log and pilot status.json.' }
    $notification = 'V42 YOLO35 crop-jitter/long512 pilot completed. Review outputs/recognition/pilots/v42_yolo35_jitter_long512_2000/analysis.md,report.json,sampling.json,yolo_views.jsonl,initial_parity.json,metrics.jsonl and paired_analysis.json. Verify2000updates/64000samples,35%YOLO22400views/all1561traincrops,30%true11–14char512lines19200,34%totalphysical512,15%newchar/20%otherreplay,random2–5%per-sidepositivepadding and.95–1.05horizontalstretch(no added opticalnoise),frozen147BNbuffers/trainableaffine. SourceV41bestEMA1000,fullfine-tune withlowLRs andfrozen-sourceKL onsyntheticreplay. Compareclean156YOLOholdout,matchedclean/augtrainprobe,long512vs384,newcharrecall,raw/EMAselected/finalandretention. Holdoutgeometry/source/registryunchanged;noautomaticpromotion. One-shot completion notification;no polling.'
} catch {
    $notification = 'V42 YOLO35 crop-jitter/long512 pilot failed. Continue authorized2000-update task by inspecting outputs/recognition/v42.stderr.log and pilots/v42_yolo35_jitter_long512_2000/status.json. Fix safely and resume;preserve V38/V41 andalloriginalcheckpoints. Userauthorized35%YOLOwith2–5%boundarypadding/slightstretchandincreased512exposure. One-shot failure notification;no polling.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v42_notification.json' -Encoding utf8
    if ($notification) { & herdr agent prompt 'w2:p1' $notification }
}
