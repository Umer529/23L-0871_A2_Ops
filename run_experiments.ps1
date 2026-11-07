# Task 3.3 / 3.5 - all VisionGuard MLflow runs (logged to DagsHub via .env).
# Each run differs from the one it is compared with in ONE change:
#   E1 baseline           mobilenetv2, 224, aug on, last block unfrozen, ft-lr 1e-5
#   E2 vs E1              fine-tune learning rate 1e-5 -> 1e-4
#   E3 vs E2              unfrozen layers: last block -> last 50 layers
#   E4 vs E3              augmentation off
#   E5 vs E3              image size 224 -> 160
#   E6 vs E3              backbone MobileNetV2 -> EfficientNetB0
#   E7                    baseline CNN with MLflow autologging (Task 3.3 second approach)
# Usage (from the repo root, venv active):  .\run_experiments.ps1
# "Continue": TensorFlow logs to stderr, which PowerShell 5.1 would otherwise treat as fatal;
# real failures are caught through $LASTEXITCODE below
$ErrorActionPreference = "Continue"
$py = if (Test-Path .\venv\Scripts\python.exe) { ".\venv\Scripts\python.exe" } else { "python" }
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$common = @("--tracking", "manual", "--epochs", "12", "--ft-epochs", "10")
New-Item -ItemType Directory -Force logs | Out-Null

$runs = [ordered]@{
  "E1_mnv2_base"         = @("--model", "mobilenetv2")
  "E2_mnv2_ftlr1e-4"     = @("--model", "mobilenetv2", "--ft-lr", "1e-4")
  "E3_mnv2_unfreeze50"   = @("--model", "mobilenetv2", "--ft-lr", "1e-4", "--unfreeze", "50")
  "E4_mnv2_unfreeze50_noaug" = @("--model", "mobilenetv2", "--ft-lr", "1e-4", "--unfreeze", "50", "--no-aug")
  "E5_mnv2_unfreeze50_160"   = @("--model", "mobilenetv2", "--ft-lr", "1e-4", "--unfreeze", "50", "--img-size", "160")
  "E6_effb0_unfreeze50"  = @("--model", "efficientnetb0", "--ft-lr", "1e-4", "--unfreeze", "50")
}
foreach ($name in $runs.Keys) {
  Write-Host "=== $name ===" -ForegroundColor Cyan
  & $py src/train.py @($runs[$name]) @common --run-name $name 2>&1 | Out-File -Encoding utf8 "logs/$name.log"
  if ($LASTEXITCODE -ne 0) { throw "$name failed - see logs/$name.log" }
}
Write-Host "=== E7_cnn_autolog ===" -ForegroundColor Cyan
& $py src/train.py --model cnn --tracking autolog --epochs 20 --run-name E7_cnn_autolog 2>&1 | Out-File -Encoding utf8 logs/E7_cnn_autolog.log
Write-Host "All runs done. Next: python src/register.py" -ForegroundColor Green
