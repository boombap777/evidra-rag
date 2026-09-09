param(
    [ValidateSet("dashboard", "mcp", "embedding")]
    [string]$Mode = "dashboard",
    [int]$DashboardPort = 8503,
    [switch]$Demo,
    [string]$LlamaServer = $env:LLAMA_SERVER_PATH,
    [string]$EmbeddingModel = $env:RAG_EMBEDDING_MODEL_PATH
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $PSScriptRoot "logs"

if ($Demo) {
    if ($Mode -ne "dashboard") { throw "-Demo supports dashboard mode only." }
    & $python scripts\start_dashboard.py --demo --port $DashboardPort
    exit $LASTEXITCODE
}
if (-not $LlamaServer -or -not $EmbeddingModel) {
    throw "Use -Demo, or supply -LlamaServer and -EmbeddingModel (also available via LLAMA_SERVER_PATH and RAG_EMBEDDING_MODEL_PATH)."
}

foreach ($required in @($llamaServer, $embeddingModel, $python)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required local runtime file is missing: $required"
    }
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot "data\db") -Force | Out-Null

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Wait-LocalHealth {
    param(
        [string]$Uri,
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            throw "Local embedding process exited before becoming healthy: $Uri"
        }
        try {
            $response = Invoke-RestMethod -Uri $Uri -TimeoutSec 2
            if ($response.status -eq "ok") {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Timed out waiting for local embedding service: $Uri"
}

$embeddingArguments = @(
    "-m", ('"' + $embeddingModel + '"'),
    "--host", "127.0.0.1",
    "--port", "18282",
    "--embedding",
    "--pooling", "last",
    "--n-gpu-layers", "99",
    "--ctx-size", "4096",
    "--batch-size", "512",
    "--ubatch-size", "512",
    "--no-ui"
)

$embeddingProcess = $null

try {
    $embeddingProcess = Start-Process -FilePath $llamaServer -ArgumentList $embeddingArguments `
        -RedirectStandardOutput (Join-Path $logDir "embedding-model.out.log") `
        -RedirectStandardError (Join-Path $logDir "embedding-model.err.log") `
        -PassThru -WindowStyle Hidden

    Wait-LocalHealth -Uri "http://127.0.0.1:18282/health" -Process $embeddingProcess

    switch ($Mode) {
        "dashboard" {
            & $python scripts\start_dashboard.py --host 127.0.0.1 --port $DashboardPort
        }
        "mcp" {
            & $python -m src.mcp_server.server
        }
        "embedding" {
            Write-Host "Embedding API: http://127.0.0.1:18282/v1"
            Wait-Process -Id $embeddingProcess.Id
        }
    }
}
finally {
    if ($null -ne $embeddingProcess -and -not $embeddingProcess.HasExited) {
        Stop-Process -Id $embeddingProcess.Id -Force
        $embeddingProcess.WaitForExit()
    }
}
