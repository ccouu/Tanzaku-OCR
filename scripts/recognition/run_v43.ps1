$ErrorActionPreference = 'Stop'
$workspace = 'D:\profolio\japanese_translation'
Set-Location -LiteralPath $workspace
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = $workspace
$python = Join-Path $workspace '.venv-recognition\Scripts\python.exe'
$notification = ''
try {
    & $python '-m' 'scripts.recognition.train_v43_digits' '--resume'
    if ($LASTEXITCODE -ne 0) { throw 'V43 failed; inspect v43.stderr.log and pilot status.json.' }
    $notification = 'V43 digit/price2000-update pilot completed. Review outputs/recognition/pilots/v43_digits_prices_2000/analysis.md,report.json,sampling.json,initial_parity.json,metrics.jsonl and paired_analysis.json. Verify64000samples:30%digit/25%YOLO/20%long512/10%newcharacter/15%ordinaryreplay; balanced10digitanchors/all6000trainingrenders; full-width digits normalized to ASCII in every training stream with original source labels preserved; 1000new-seed probes including4heldoutfonts; all1561YOLOtrain and521exactterms;147fixedBNbuffers. Compare sourceV42raw2000 versus raw/EMA on digitrecall/D-I-S/repeateddigits/prices and clean156YOLOholdout/long512/retention. InitialV42 fallback is NOT gate eligible: distinguish best.pt from final raw/EMA. ProductionV38unchanged,noautopromotion. One-shot completion notification;no polling.'
} catch {
    $notification = 'V43 digit/price pilot failed. Inspect outputs/recognition/v43.stderr.log and pilots/v43_digits_prices_2000/status.json; fix safely and resume original authorized2000-update task. Preserve all V42/source checkpoints and production registry. Full-width digit labels must normalize to ASCII. One-shot failure notification;no polling.'
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
} finally {
    @{message=$notification;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath 'outputs/recognition/v43_notification.json' -Encoding utf8
    if ($notification) { & herdr agent prompt 'w2:p1' $notification }
}
