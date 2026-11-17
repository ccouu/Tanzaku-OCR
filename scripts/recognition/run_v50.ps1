$ErrorActionPreference='Stop'
$workspace='D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE='1'
$env:PYTHONPATH=$workspace
$python=Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$root='outputs/recognition/pilots/v50_normalized_matched_2000'
$phase='normalized frozen-BiLSTM control'
$notification=''
try {
    & $python '-m' 'scripts.recognition.train_v47_v48' '--arm' 'selective' '--normalize-parentheses' '--output' "$root/control" '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V50 normalized control failed.' }
    $phase='normalized BiLSTM1e-5 arm'
    & $python '-m' 'scripts.recognition.train_v47_v48' '--arm' 'selective' '--normalize-parentheses' '--unfreeze-bilstm' '--output' "$root/bilstm" '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V50 normalized BiLSTM arm failed.' }
    $notification='V50normalizedmatchedexperimentcompleted:2000updatesEACHin outputs/recognition/pilots/v50_normalized_matched_2000/control and bilstm. Review analysis.md,report.json,run_config.json,sampling.json,initial_parity.json,metrics.jsonl andraw/EMAselected/final. BothstartindependentlyfromV43withV49exactsourcebatchplan/seed/66rowgroups/7streams:Latin1e-3,digit+bracket1e-5,100warmup,fullsoftmaxV43teacherKLvalidnonLatinreplay. OnlytreatmentdifferenceisresidualBiLSTM+outputprojectionunfrozenLR1e-5;controlBiLSTMfrozen. BOTHnormalizetrainingparentheses（→(,）→),poolwidth-pairprobabilitymassbeforeCTC/greedycollapseandatKLtemperature2;loss-onlyaliascolumncompactiondoesnotchangecheckpointIDs. Literalraw,text-normalized,andpooledcanonicalevaluationstoredseparately. Verifyzero-updateRAWV43/V49parity,normalization-onlyinitialeffects,allCNN/projection/temporal/BNaffine/147buffers/unselectedrowsfrozen,64kpresentationsperarm/all6000digit/all1561YOLO/all521terms,andcanonicaltargets/source-labelpreservation. ComparearmsdirectlynotonlyagainstliteralV49;focusnew33Latin/openingbrackets/digits/shorttext/YOLO156and17RAWretentiongatesbothprecisions. SelectedrankingusescanonicalCERbutV49rawguardsthresholdsunchanged;distinguishselectedfromfinal. V43remainspromoted,nofurthertraining,nocleanup,nopolling. One-shotcompletionnotification.'
} catch {
    $notification="V50 normalized matched experiment failed during $phase. Inspect outputs/recognition/v50.stderr.log and pilots/v50_normalized_matched_2000/control or bilstm/status.json. Fixsafelyandresumeauthorized2000updatesperarm;preserveV43/V49andallsourcecheckpoints. Do not unfreeze CNN/temporal/BN or otherclassifierrows. BotharmsmustusenormalizedparenthesissemanticsandidenticalV49presentationplan. No promotion or extra training. One-shotfailuremessage;no polling."
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;phase=$phase;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content outputs/recognition/v50_notification.json -Encoding utf8
    if ($notification) {
        & herdr agent prompt 'w2:p1' $notification
        if ($LASTEXITCODE -ne 0) { @{delivered=$false;local_message_preserved=$true} | ConvertTo-Json | Set-Content outputs/recognition/v50_notification_delivery.json -Encoding utf8 }
    }
}
