$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$work = Join-Path $workspace 'outputs\detection\training\banana50_finetune_v3'
New-Item -ItemType Directory -Force -Path $work | Out-Null
$notification = ''
try {
    $recognitionStatus = Join-Path $workspace 'outputs\recognition\pilots\v37_long512_2000\status.json'
    $recognitionProcessFile = Join-Path $workspace 'outputs\recognition\v37_process.json'
    if (Test-Path -LiteralPath $recognitionProcessFile) {
        $recognitionProcess = Get-Content -LiteralPath $recognitionProcessFile -Raw | ConvertFrom-Json
        $activeProcess = Get-Process -Id $recognitionProcess.pid -ErrorAction SilentlyContinue
        if ($activeProcess) {
            $startedUtc = $activeProcess.StartTime.ToUniversalTime()
            if ($recognitionProcess.started_utc -is [DateTime]) {
                $expectedUtc = $recognitionProcess.started_utc.ToUniversalTime()
            } else {
                $expectedUtc = [DateTimeOffset]::Parse($recognitionProcess.started_utc,[Globalization.CultureInfo]::InvariantCulture).UtcDateTime
            }
            if ([Math]::Abs(($startedUtc-$expectedUtc).TotalSeconds) -gt 10) { throw 'Recognition PID identity changed; refusing to wait on unrelated process.' }
            @{state='waiting_for_v37_gpu';waiting_pid=$recognitionProcess.pid;latest_only=$true;train_images=45;val_images=5;updated_utc=[DateTime]::UtcNow.ToString('o')} |
                ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'status.json') -Encoding utf8
            @{state='waiting_for_v37_gpu';waiting_pid=$recognitionProcess.pid;updated_utc=[DateTime]::UtcNow.ToString('o')} |
                ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'launcher_status.json') -Encoding utf8
            Wait-Process -Id $recognitionProcess.pid -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path -LiteralPath $recognitionStatus) {
        $lastRecognitionStatus=Get-Content -LiteralPath $recognitionStatus -Raw | ConvertFrom-Json
        if ($lastRecognitionStatus.state -ne 'completed') { throw 'V37 did not complete; fine-tuning remains staged to avoid conflicting GPU recovery.' }
    }
    & $python '-m' 'scripts.stage2_yolo.finetune_banana_latest'
    if ($LASTEXITCODE -ne 0) { throw 'YOLOlatest-onlyfine-tuningfailed;inspectbanana50_finetune_v3logs/status.' }
    $notification='Latest-annotation-only YOLO fine-tuning completed. Review outputs/detection/training/banana50_finetune_v3/evaluation_report.json,status.json,runs/menu_obb/results.csv and data/datasets/detection/banana50_latest_v3/training_manifest.json. Verify45train/5heldoutmenus,all50latestreviewedpages/668boxes,noolderannotationdata,andcomparecheckpointbaselineversusnewbest. Existingcheckpoints/yolo11n_menu_obb_best.ptunchanged;newcandidateissavedseparately. One-shot Herdr completion notification;no polling.'
} catch {
    @{state='failed';error=$_.Exception.Message;updated_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'launcher_status.json') -Encoding utf8
    $notification='Latest-onlyYOLOfine-tuninglauncherfailedorstageddependencyfailed. Inspect outputs/detection/training/banana50_finetune_v3/launcher_status.json,status.json andstderr.log. Userauthorizedfine-tuningonlylatest50reviewedmenusfromexistingcheckpoint. Fixsafelyandcontinue;do notinterruptV37oroverwriteoriginalweights. One-shot failure notification.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    if ($notification) {
        & herdr agent prompt 'w2:p1' $notification
        @{exit_code=$LASTEXITCODE;message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'notification.json') -Encoding utf8
    }
}
