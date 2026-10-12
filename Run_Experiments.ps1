param([switch]$ResumeBert)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONHASHSEED = '42'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '请先按照 README 安装环境。' }
& $pythonExe scripts/download_assets.py
if ($LASTEXITCODE -ne 0) { throw '模型下载失败。' }
& $pythonExe src/train_classical.py --methods binary frequency tfidf glove w2v_ag w2v_nyt
if ($LASTEXITCODE -ne 0) { throw '传统模型训练失败。' }
$bertArgs = @('src/train_bert.py','--model','cache/bert-base-uncased','--threads','6')
if ($ResumeBert) { $bertArgs += '--resume' }
& $pythonExe @bertArgs
if ($LASTEXITCODE -ne 0) { throw 'BERT 训练失败。' }
& $pythonExe src/project_analysis.py
if ($LASTEXITCODE -ne 0) { throw '大作业实验失败。' }
& $pythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw '系统测试失败。' }
Write-Host '实验完成。使用 scripts/build_pdfs.py 和 scripts/build_acl.py 更新报告。'
