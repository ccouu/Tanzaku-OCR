$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
try {
    & $python '-m' 'scripts.recognition.train_v40_new_characters' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V40 failed; inspect v40.stderr.log and training status.json.' }
    $notification = 'V40 expanded-vocabulary2000-update training completed. Review outputs/recognition/training/v40_new_characters_2000/analysis.md,report.json,sampling.json,new_character_coverage.json,new_row_warmup_audit.json,initial_parity.json,metrics.jsonl and paired_analysis.json. Verify1722->1779characters(blank0unchanged),all57newIDsappended,2000updates/64000presentations,50%targeted/25%YOLO/25%synthetic,actualnew-characterencounters(minimum532planned),all57newclassifierrowsupdated,147fixedBNbuffers,andfirst200updatesonlynewrows. Compare initializedV38/newvocab,selectedbestandfinalraw/EMA on new-character recall,expanded156-crop YOLOholdout and revisedretention. V38/sourcevocabulary/productionregistryunchanged;noautomaticpromotion. One-shot completion notification;no polling.'
} catch {
    $notification = 'V40 expanded-vocabulary training failed. Continue authorized2000-update task by inspecting outputs/recognition/v40.stderr.log and training/v40_new_characters_2000/status.json. Fix safely and resume; preserve promotedV38,originalvocabularies,andallcheckpoints. All57newcharactersmustreceivemultiplesupervisedencounters. One-shot failure notification.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v40_notification.json' -Encoding utf8
    if ($notification) { & herdr agent prompt 'w2:p1' $notification }
}
