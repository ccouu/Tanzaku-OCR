$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
try {
    & $python '-m' 'scripts.recognition.train_v44_adaptation' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V44 failed; inspect v44.stderr.log and pilot status.json.' }
    $notification = 'V44 Latin/parentheses/extra-long2000-update pilot completed. Review outputs/recognition/pilots/v44_latin_parentheses_long_2000/analysis.md,report.json,sampling.json,geometry_preflight.json,initial_parity.json,metrics.jsonl and paired_analysis.json. Verify V43finalraw source,33appendedLatinletters(allA-Z/a-z,oldIDs/blankunchanged),2000updates/64000samples,30%YOLO/all1561traincrops,15%Latin/5%dedicatedparentheses/15%digits/15%16-24charlong/5%oldlong/5%oldnewchar/10%ordinaryreplay,521exactterms,full-widthdigitlabelnormalization,147fixedBNbuffers. Geometrypreflightcomparedfit512/wide768/overlap512with100normalized-pixeloverlap; synthetic-heldout-font choice and realholdoutreported separately. Verifyfirst200updatesonlynewclassifierrows; compare raw/EMA on Latincoverage/parentheses/digits/longwideversusoverlap/156YOLOholdout/retention. InitialV43fallbackisnoteligible; distinguishbest.ptfromfinalraw/EMA. V38/registry/sourceunchanged,noautopromotion. One-shot completion notification;no polling.'
} catch {
    $notification = 'V44 Latin/parentheses/extra-long pilot failed. Inspect outputs/recognition/v44.stderr.log and pilots/v44_latin_parentheses_long_2000/status.json; fix safely and resume authorized2000-update task. User chose V43finalraw,30%YOLO,Latincoverage,parenthesesandlonglines. Preserve all source checkpoints/IDs and registry. One-shot failure notification;no polling.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v44_notification.json' -Encoding utf8
    if ($notification) { & herdr agent prompt 'w2:p1' $notification }
}
