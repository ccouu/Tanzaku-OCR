$ErrorActionPreference='Stop'
$workspace='D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE='1'
$env:PYTHONPATH=$workspace
$python=Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$phase='Latin fitting'
$notification=''
try {
    & $python '-m' 'scripts.recognition.train_v47_v48' '--arm' 'latin' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V47 Latin fitting failed.' }
    $phase='independent parentheses arm'
    & $python '-m' 'scripts.recognition.train_v47_v48' '--arm' 'parentheses' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V48 parentheses arm failed.' }
    $notification='V47 tiny clean Latin fitting1000updates and V48 independent parentheses2000updates completed. Review outputs/recognition/pilots/v47_clean_latin_fit_1000/ and v48_parentheses_2000/analysis.md,report.json,sampling.json,initial_parity.json,metrics.jsonl and final/raw/EMA/best checkpoints. V47uses208cleanwhole-stringNotoSansrenders/all52trainableLatinrows+BiLSTM,noKL/replay;bestisfittingonlynotretentionqualified;comparetrainmemorization,samefontnewlayout,4unseen-fitfonts,single/repeated/word,new33/old19andD-I-S. V48startsindependentlyfromV43with1779vocab;30%balancedASCII/fullwidthparentheses,45%synthetic,25%YOLO;trainBiLSTM/head,frozenCNN/temporal/BNaffine/147buffers,fullsoftmaxV43teacherKLreplay. CompareexistingV44bracketprobes/newbalancedprobes/156YOLOholdout/digits/17gatesbothprecisions; distinguishselectedfromfinal. V43remainspromoted,nofurthertraining,nocleanup,nopolling. One-shotcompletionnotification.'
} catch {
    $notification="Isolated V47/V48 experiments failed during $phase. Inspect outputs/recognition/v47_v48.stderr.log and the respective pilot status.json. Fix safely and resume the authorized1000Latin/2000parenthesis tasks independently from V43; preserve all source/current checkpoints. No promotion or extra architecture unfreezing. One-shotfailuremessage;no polling."
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;phase=$phase;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content outputs/recognition/v47_v48_notification.json -Encoding utf8
    if ($notification) {
        & herdr agent prompt 'w2:p1' $notification
        if ($LASTEXITCODE -ne 0) { @{delivered=$false;local_message_preserved=$true} | ConvertTo-Json | Set-Content outputs/recognition/v47_v48_notification_delivery.json -Encoding utf8 }
    }
}
