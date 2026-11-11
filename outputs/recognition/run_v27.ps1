$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:\profolio\japanese_translation'
$env:HF_HUB_OFFLINE='1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'
$env:PYTHONUNBUFFERED='1'
$taskPython='D:\profolio\japanese_translation\.venv-recognition\Scripts\python.exe'
for ($attempt=1; $attempt -le 3; $attempt++) {
    & $taskPython -u -m src.recognition.font_training --epochs 20
    if ($LASTEXITCODE -eq 0) { exit 0 }
    if ($attempt -lt 3) { Start-Sleep -Seconds 10 }
}
exit $LASTEXITCODE
