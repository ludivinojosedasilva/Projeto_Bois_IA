# Powershell helper: instala dependências (se necessário), builda e inicia o servidor de produção
Set-StrictMode -Version Latest
$cwd = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Push-Location $cwd
try {
    Write-Output "Instalando dependências (se já estiverem instaladas, ignora)..."
    npm install
    Write-Output "Executando build e iniciando servidor de produção..."
    npm run start:prod
} finally {
    Pop-Location
}
