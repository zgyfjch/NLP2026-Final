$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '请先按照 README 安装 Python 环境。' }
try {
    $health = Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 2
} catch {
    Start-Process -FilePath $pythonExe -ArgumentList 'src/demo.py' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'demo.log') -RedirectStandardError (Join-Path $PSScriptRoot 'demo-error.log')
    for ($attempt=0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try { $health = Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 2; break } catch {}
    }
}
if ($health.status -ne 'ok') { throw 'Demo 启动失败，请查看 demo-error.log。' }
Start-Process 'http://127.0.0.1:8765'
