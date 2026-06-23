**BovinoVision — Sistema de Análise da Condição Corporal de Bovinos**

Bem-vindo ao repositório do BovinoVision. Este README descreve de forma completa o que o projeto contém, as dependências usadas, como executar localmente, como gerar o build de produção e como configurar deploy automático via GitHub Actions para um servidor Linux com systemd.

**Visão Geral**:
- **Propósito**: Ferramenta web para avaliar condição corporal de bovinos, com interface React + backend Node/Express.
- **Arquitetura**: SPA React (Vite) servido por um servidor Node que também implementa rotas de API e integrações (Firebase Admin, Supabase, serviços de email).

**Estrutura do Projeto (resumo)**:
- `src/` — código cliente React (componentes, estilos, assets).
- `server.ts` — servidor Node/Express (entry server para dev e build server.cjs para produção).
- `server/` — helpers de servidor (ex.: `emailService.ts`).
- `lib/` — integrações (Firebase, Supabase, schemas).
- `.github/workflows/` — workflow de CI/CD (deploy automático).
- `dist/` — artefatos de produção gerados por `npm run build`.

**Tecnologias e Dependências Principais**
- Frontend: `react`, `react-dom`, `@vitejs/plugin-react`, `vite`.
- Backend: `express`, `node` (via `server.ts`, empacotado com `esbuild`).
- Autenticação / BaaS: `firebase`, `firebase-admin`, `@supabase/supabase-js`.
- Utilitários: `zod` (validação), `nodemailer` (envio de email), `dotenv`.
- Ferramentas de build: `vite`, `esbuild`, `tsx` (dev server), `typescript`.

Confira o `package.json` para a lista completa de dependências e versões.

**Scripts úteis (package.json)**
- `npm run dev` — inicia o servidor em modo desenvolvimento usando `tsx server.ts`.
- `npm run build` — executa `vite build` e empacota `server.ts` em `dist/server.cjs` com `esbuild`.
- `npm run start` — executa `node dist/server.cjs` (assume que `dist` já foi gerado).
- `npm run start:prod` — (helper) roda `npm run build && npm run start`.

**Variáveis de ambiente**
- Use um arquivo `.env` (não comitado) para configurar chaves e segredos:
  - `PORT` — porta do servidor (padrão 3000)
  - `FIREBASE_*` — credenciais do Firebase (conforme `lib/firebase-admin.ts`)
  - `SUPABASE_URL` / `SUPABASE_KEY` — credenciais Supabase
  - `SMTP_*` — credenciais do servidor SMTP para `nodemailer`
  - Outras chaves específicas usadas no código (ver `lib` e `server.ts`).

**Executar localmente (desenvolvimento)**
1. Instale Node.js LTS (recomendado) e git.
2. No terminal, na raiz do projeto:
```
cd "C:\Users\Pedro\Downloads\BovinoVision AI-System"
npm install
```
3. Crie seu `.env` com as variáveis necessárias (baseie-se em `.env.example` se existir).
4. Inicie em modo dev:
```
npm run dev
```
5. Abra `http://localhost:3000` no navegador.

**Build e execução em produção (local ou servidor)**
1. Gerar build e empacotar o server:
```
npm run build
```
2. Iniciar servidor de produção (assume `node` instalado no host):
```
npm run start
```
Ou execute o helper criado (PowerShell / CMD):
```
.\run-prod.ps1    # PowerShell
run-prod.bat      # CMD
```

**Deploy automático via GitHub Actions (fluxo usado aqui)**
- O workflow empacota `dist/`, copia para o servidor via SCP usando uma chave SSH armazenada em GitHub Secrets, descompacta, instala dependências de produção e reinicia o serviço `systemd` (`bovinovision`).

Secrets GitHub recomendados (Repository → Settings → Secrets → Actions):
- `SSH_PRIVATE_KEY` — chave privada criada para o pipeline (não compartilhe).
- `SSH_HOST` — host ou IP do servidor.
- `SSH_USER` — usuário SSH no servidor.
- `SSH_PORT` — opcional (22 por padrão).
- `REMOTE_DIR` — diretório remoto onde os arquivos serão extraídos (ex.: `/var/www/bovino`).

**Configuração do servidor (ex.: Ubuntu VPS)**
1. Preparar diretório remoto e Node:
```
sudo mkdir -p /var/www/bovino
sudo chown -R $USER:$USER /var/www/bovino
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```
2. Adicionar chave pública (`~/.ssh/authorized_keys`) para o usuário do deploy.
3. Criar unit `systemd` (exemplo):
```
sudo tee /etc/systemd/system/bovinovision.service > /dev/null <<'EOF'
[Unit]
Description=BovinoVision Node app
After=network.target

[Service]
User=SEU_USUARIO
WorkingDirectory=/var/www/bovino
ExecStart=/usr/bin/node /var/www/bovino/dist/server.cjs
Restart=always
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bovinovision
sudo systemctl start bovinovision
sudo systemctl status bovinovision
```

**Como funciona o workflow de deploy**
- Ao dar push para `main`, o GitHub Actions executa os passos:
  1. Checkout do código
  2. `npm ci` e `npm run build`
  3. Cria um `release.tar.gz` contendo `dist` e arquivos essenciais
  4. Copia o arquivo para o servidor via SCP
  5. No servidor: descompacta, instala `npm install --production` e reinicia o `systemd` service

Se quiser limitar branches ou adicionar checks (tests, lint), atualize `.github/workflows/deploy.yml`.

**Dicas de troubleshooting**
- Se `npm install` falhar por permissões no Windows, rode o terminal como administrador.
- Se o servidor retornar 500s, verifique logs:
```
sudo journalctl -u bovinovision -f
```
- Para problemas com secrets ou SCP, verifique se a chave privada no GitHub Secrets corresponde à chave pública instalada em `~/.ssh/authorized_keys` do servidor.

**Contribuição e testes**
- Sinta-se à vontade para abrir issues no repositório com bugs, melhorias ou dúvidas.
- Para desenvolvimento, crie branches com nomes descritivos e abra PRs para `main`.

Pedro Omna

---
Arquivo gerado automaticamente com instruções abrangentes para deploy e execução local.
