# Respaldo: arranca JupyterLab en el navegador con el env sales_env.
# Uso: clic derecho > "Ejecutar con PowerShell", o desde una terminal:  .\start_jupyter.ps1
# Sirve cuando el editor de notebooks de VS Code se cuelga (bug del webview).
$ErrorActionPreference = "Stop"

# Correr siempre desde la raíz del proyecto (para que los imports de src/ funcionen)
Set-Location $PSScriptRoot

# Activar el env de Miniforge/Mamba
& "C:\Users\Makora.LOUMAK\miniforge3\shell\condabin\conda-hook.ps1"
conda activate sales_env

Write-Host "JupyterLab arrancando en http://localhost:8888  (Ctrl+C para detener)" -ForegroundColor Green
jupyter lab
