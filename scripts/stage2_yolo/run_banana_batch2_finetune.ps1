$ErrorActionPreference='Stop'
$workspace='D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:PYTHONPATH=$workspace
$python=Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$work=Join-Path $workspace 'outputs\detection\training\banana51_100_finetune_v4'
$notification=''
try {
    & $python '-m' 'scripts.stage2_yolo.finetune_banana_batch2'
    if ($LASTEXITCODE -ne 0) { throw 'Batch2YOLOtrainingfailed;inspectv4status/stderr.' }
    $notification='YOLO batch2fine-tuning completed. Review outputs/detection/training/banana51_100_finetune_v4/evaluation_report.json,status.json,runs/menu_obb/results.csv and data/datasets/detection/banana51_100_v4/training_manifest.json. Verifytrainingonly45latestmenus051-100,validation5new+5previousmenus,reviewedlabels/hash/splitintegrity,sourcecheckpointunchanged. Comparecombined/latest/previousvalidationbaselinevsnewbest;assessforgettingonpreviousval. No additionaltrainingrunning. One-shot Herdrcompletionnotification;no polling.'
} catch {
    $notification='YOLO batch2fine-tuning failed. Inspect outputs/detection/training/banana51_100_finetune_v4/status.json andstderr.log. Userauthorizedlatestlabelswithpreviousvalidationset;fixsafelyandcontinuewithoutoverwritingpriorweights. One-shot failuremessage.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    if ($notification) {
        & herdr agent prompt 'w2:p1' $notification
        @{message=$notification;exit_code=$LASTEXITCODE;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path $work 'notification.json') -Encoding utf8
    }
}
