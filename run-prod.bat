@echo off
REM Batch helper: instala dependências, builda e inicia o servidor de produção
cd /d %~dp0
echo Instalando dependencias (se ja estiverem instaladas, ignora)...
npm install
echo Executando build e iniciando servidor de producao...
npm run start:prod
