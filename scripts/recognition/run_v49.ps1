$ErrorActionPreference='Stop'
$workspace='D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE='1'
$env:PYTHONPATH=$workspace
$python=Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification=''
try {
    & $python '-m' 'scripts.recognition.train_v47_v48' '--arm' 'selective' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V49 selective head-only pilot failed.' }
    $notification='V49selectivehead-only2000updatepilotcompleted. Review outputs/recognition/pilots/v49_selective_head_2000/analysis.md,report.json,sampling.json,initial_parity.json,metrics.jsonl andraw/EMAselected/final. Verifyonly52LatinrowsLR1e-3and10digit+4parenthesisrowsLR1e-5updated; blank/allotherclassifierrows/BiLSTM/CNN/projection/temporal/BNaffine/147buffersbit-identicaltoV43. Append33LatinIDs,old1779IDsunchanged. Verify64000presentations:30Latin(15tiny/15multifont),20digit,10brackets,20YOLO,20synthetic;all6000digitimages/all1561YOLO/all521terms;balanced3200ASCII/3200fullwidthparenthesespresentations. FullsoftmaxV43teacherKLweight1T2validframesonnonLatin-labeledreplayonly. CompareV47tinyfit/generalization,V45/V46multifontLatin,digits/brackets,YOLO156holdoutand17gatesbothprecisions. Bestissource-relativegated,noproductionpromotion. V43unchanged,nofurthertraining,nocleanup,nopolling. One-shotcompletionnotification.'
} catch {
    $notification='V49selectivehead-onlypilotfailed. Inspect outputs/recognition/v49.stderr.log and pilots/v49_selective_head_2000/status.json. Fixsafelyandresumeauthorized2000updates;donotunfreezeBiLSTMorotherfeatures. PreserveV43/V38andallsourcecheckpoints. Only52Latin+10digit+4bracketrowsmaytrain. One-shotfailuremessage;nopolling.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content outputs/recognition/v49_notification.json -Encoding utf8
    if ($notification) {
        & herdr agent prompt 'w2:p1' $notification
        if ($LASTEXITCODE -ne 0) { @{delivered=$false;local_message_preserved=$true} | ConvertTo-Json | Set-Content outputs/recognition/v49_notification_delivery.json -Encoding utf8 }
    }
}
