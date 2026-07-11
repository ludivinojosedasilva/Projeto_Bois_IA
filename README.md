# Rayvora Vision Pro — Sistema Inteligente de Estimativa de Peso Bovino

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![React](https://img.shields.io/badge/React-18%20%2B%20Vite-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)
![Node.js](https://img.shields.io/badge/Node.js-18+-339933?logo=nodedotjs)
![TinyML: Edge Impulse](https://img.shields.io/badge/TinyML-Edge%20Impulse-1BA94C.svg)
![PWA](https://img.shields.io/badge/PWA-Suportado-orange?logo=pwa)
![Status: Em Desenvolvimento](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)

**Autores:** Ludivino José Da Silva, Lucas Teixeira Belli e Pedro Omna
**Disciplina:** Projeto Integrador I (2026.1) – Engenharia de Computação (UFSC)

---

## Sobre o Projeto

O **Rayvora Vision Pro** é um ecossistema inteligente de pecuária de precisão focado na estimativa de peso de bovinos a partir de imagens. Desenvolvido como parte do Projeto Integrador da UFSC, o objetivo é oferecer ao produtor rural uma alternativa móvel, de baixo custo e não invasiva à pesagem mecânica tradicional — utilizando apenas a câmera de um smartphone e modelos avançados de **Machine Learning (TinyML)**.

A arquitetura do projeto evoluiu de um protótipo inicial em Streamlit (usado durante os Sprints de desenvolvimento e validação dos modelos de IA) para uma aplicação robusta baseada em **Progressive Web App (PWA)** com React e Vite. O sistema é dividido em três camadas principais:

1. **Frontend PWA (React + Vite + TypeScript):** Interface moderna, instalável no celular do produtor e otimizada para uso no campo.
2. **Backend de IA (`ml-service` em Python / FastAPI):** API assíncrona dedicada a carregar o motor de inferência LiteRT e executar a visão computacional.
3. **Persistência em Nuvem (Firebase / Supabase):** Sincronização de dados e autenticação de usuários para manter o histórico de pesagens em tempo real.

---

## Arquitetura e Pipeline de IA

O projeto segue um fluxo distribuído em módulos independentes:

```
[ Celular / PWA ]
       │
       │  1. Usuário tira ou envia foto do bovino
       ▼
[ Backend API Python — porta 8001 ]
       │
       │  2. YOLOv8s-Seg isola o animal na imagem (modelo_bois.pt)
       │  3. LiteRT / MobileNet calcula o peso (model.tflite)
       ▼
[ Resposta: Peso (kg) + Confiança + Margem de Erro + Status Abate ]
       │
       ▼
[ Frontend PWA — exibe resultado e salva no Firebase/Supabase ]
```

### Módulos de IA

**1. Módulo de Segmentação (YOLOv8)**
Antes da estimativa de peso, a rede YOLOv8s-Seg identifica e isola o contorno do animal na imagem, removendo ruído de fundo (pastagem, cercas, outros animais). Isso padroniza a entrada para os modelos de regressão e aumenta significativamente a robustez do sistema em condições de campo.

**2. Módulo de Estimativa via Edge Impulse (Produção)**
Modelo de regressão treinado na plataforma Edge Impulse usando **Transfer Learning** sobre a arquitetura **MobileNet**. É o motor de IA atualmente em produção. Resultado: **MAE de 32.87 kg**.

**3. Módulo de Estimativa via Fine-Tuning (Experimental)**
Frente de pesquisa paralela em que um modelo MobileNet pré-treinado (obtido de um projeto de referência de outro TCC) passa por **Fine-Tuning** com o dataset próprio da equipe. Resultado atual: **MAE de 47.82 kg** — ainda não supera o modelo de produção, mas está em refinamento contínuo.

Cada módulo possui documentação técnica detalhada na sua respectiva pasta em `models/`.

---

## Funcionalidades

- **Captura Nativa e Upload de Imagem:** Otimizado para fotografar o bovino diretamente do curral pelo navegador do celular.
- **Instalação como App (PWA):** Pode ser adicionado à tela inicial do Android ou iOS como app nativo, com ícone próprio e modo tela cheia.
- **Estimativa de Peso com IA:** Backend processa a imagem com YOLOv8 (segmentação) + LiteRT (regressão), retornando peso estimado, confiança e margem de erro.
- **Status de Aptidão para Abate:** Classifica automaticamente o animal como **APTO** (peso ≥ 380 kg) ou **NÃO APTO** (peso < 380 kg).
- **Autenticação de Usuários:** Sistema de login seguro para controle de operadores da fazenda.
- **Histórico Sincronizado em Nuvem:** Pesagens salvas em Firebase Firestore / Supabase, acessíveis de qualquer dispositivo.
- **Análise de Evolução:** Painéis e gráficos para acompanhar o Ganho Médio Diário (GMD) e a evolução de peso de cada animal.

---

## Requisitos de Hardware e Software

### Hardware
- Computador para desenvolvimento (testado em Windows 11).
- Smartphone com Chrome (Android) ou Safari (iOS) atualizados.
- Dispositivos na mesma rede Wi-Fi para testes locais.

### Software — Backend (Python)
- **Python:** 3.12+
- **Gerenciador de pacotes:** `pip`

### Software — Frontend (Node.js)
- **Node.js:** 18.x ou superior
- **Gerenciador de pacotes:** `npm`

### Bibliotecas Python — Backend (`ml-service/requirements.txt`)

| Biblioteca | Função no projeto |
|---|---|
| `fastapi` | Framework da API REST assíncrona do backend de IA |
| `uvicorn` | Servidor ASGI para executar a API FastAPI |
| `ai-edge-litert` | Motor de inferência portátil (LiteRT/TFLite) para executar o modelo `.tflite` |
| `opencv-python-headless` | Processamento digital de imagens (redimensionamento, conversão de espaço de cor) |
| `numpy` | Operações numéricas e manipulação de arrays (normalização de pixels, estatísticas) |
| `pillow` | Abertura, conversão e manipulação de imagens enviadas pelo usuário |
| `python-multipart` | Suporte ao recebimento de imagens via `multipart/form-data` na API |
| `ultralytics` | Biblioteca do YOLOv8 para segmentação do bovino na imagem |

### Bibliotecas Node.js — Frontend (`package.json`)

| Biblioteca | Função no projeto |
|---|---|
| `react` + `react-dom` | Framework principal do frontend |
| `vite` | Bundler e servidor de desenvolvimento ultrarrápido |
| `typescript` | Tipagem estática do código JavaScript |
| `firebase` | SDK do Firebase para autenticação e Firestore |
| `@supabase/supabase-js` | SDK do Supabase para banco de dados relacional |
| `vite-plugin-pwa` | Plugin para transformar o app em PWA (manifest, service worker) |

---

## Estrutura do Repositório

O repositório está organizado em duas partes principais: os **recursos de IA** (modelos treinados, datasets, protótipo de validação) na raiz, e o **app PWA** (frontend + backend) dentro de `desenvolvimento-apk/`.

```
Projeto_Bois_IA/
│
│   README.md
│   requirements.txt          # Dependências Python do protótipo Streamlit
│   .gitignore
│   LICENSE
│
├── src/                      # 🔬 Protótipo Streamlit (usado nos Sprints iniciais de validação)
│   ├── app.py                # Interface Streamlit para validação dos modelos de IA
│   └── main.py               # Script auxiliar
│
├── models/                   # 🤖 Modelos de IA treinados e documentados
│   ├── segmentation/         # YOLOv8s-Seg — mAP50: 0.9948 / mAP50-95: 0.9735
│   │       modelo_bois.pt    # Modelo PyTorch (recomendado)
│   │       modelo_bois.onnx  # Formato multiplataforma
│   │       best_float32.tflite
│   │       best_float16.tflite
│   │       segmentacao_bois_yolov8.ipynb
│   │       README.md
│   │
│   ├── edge-impulse/         # Transfer Learning (MobileNet) — MAE: 32.87 kg ✅ PRODUÇÃO
│   │       model.tflite      # Modelo em produção (usado pelo ml-service)
│   │       edge-impulse-export.zip
│   │       README.md
│   │
│   └── fine-tuning/          # Fine-Tuning experimental — MAE: 47.82 kg 🧪
│           model.tflite
│           FineTuning_Rayvora.ipynb
│           README.md
│
├── data/
│   └── dataset/              # Dataset de imagens + pesos reais
│           README.md         # ⚠️ Arquivo grande (2.5 GB) — fora do Git, ver link no README
│
├── external/
│   └── TCC_V2-master.zip     # Projeto de referência usado como base do Fine-Tuning
│       README.md
│
└── desenvolvimento-apk/      # 📱 App PWA completo (Frontend + Backend de IA)
    │
    ├── src/                  # 💻 Frontend React (Componentes, Views, Estilos)
    │   ├── components/       # Componentes React (telas, modais, sidebar, header)
    │   ├── lib/              # Integrações (Firebase, Supabase, schemas)
    │   ├── assets/images/    # Imagens estáticas do app
    │   ├── App.tsx           # Componente raiz
    │   ├── main.tsx          # Entry point do React
    │   └── index.css         # Estilos globais
    │
    ├── ml-service/           # 🧠 Backend Python (API FastAPI de IA)
    │   ├── main.py           # API com endpoint /predict
    │   ├── requirements.txt  # Dependências Python do backend
    │   └── models/           # ⚠️ PASTA NÃO VERSIONADA — deve ser criada localmente
    │           modelo_bois.pt     # OBRIGATÓRIO — copiar de models/segmentation/
    │           model.tflite       # OBRIGATÓRIO — escolher UM dos dois abaixo:
    │                              #   Opção A (recomendada): models/edge-impulse/model.tflite
    │                              #   Opção B (experimental): models/fine-tuning/model.tflite
    │
    ├── public/               # Ícones PWA, favicon, service worker
    ├── supabase/             # Schema e migrations do banco relacional
    ├── Prototipo-das-telas/  # Mockups e protótipos de interface
    ├── firebase-applet-config.json
    ├── firestore.rules       # Regras de segurança do Firestore
    ├── package.json          # Dependências Node.js
    ├── vite.config.ts        # Configuração do Vite e PWA
    └── tsconfig.json
```

> **Nota:** A pasta `desenvolvimento-apk/ml-service/models/` **não está versionada no Git** (está no `.gitignore`) porque contém arquivos binários pesados. Ela deve ser criada manualmente seguindo as instruções abaixo.

---

## Instalação e Configuração

### Passo 1: Clonar o repositório

```sh
git clone https://github.com/ludivinojosedasilva/Projeto_Bois_IA.git
cd Projeto_Bois_IA
```

### Passo 2: Configurar os modelos do Backend de IA

A pasta `desenvolvimento-apk/ml-service/models/` precisa ser criada manualmente e populada com os modelos corretos:

```sh
# Criar a pasta de modelos do backend
mkdir desenvolvimento-apk/ml-service/models
```

Copie os seguintes arquivos para dentro dessa pasta:

| Arquivo | Origem no repositório | Obrigatoriedade |
|---|---|---|
| `modelo_bois.pt` | `models/segmentation/modelo_bois.pt` | ✅ Obrigatório (segmentação YOLOv8) |
| `model.tflite` | `models/edge-impulse/model.tflite` *(recomendado)* **ou** `models/fine-tuning/model.tflite` | ✅ Obrigatório (escolha um) |

> **Qual `.tflite` usar?**
> - **`models/edge-impulse/model.tflite`** → Transfer Learning (MobileNet), MAE 32.87 kg. **Recomendado para produção.**
> - **`models/fine-tuning/model.tflite`** → Fine-Tuning experimental, MAE 47.82 kg. Use apenas para testes e pesquisa.

### Passo 3: Configurar o Backend de IA

```sh
cd desenvolvimento-apk/ml-service

# Criar e ativar ambiente virtual
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Iniciar a API na porta 8001
uvicorn main:app --host 0.0.0.0 --port 8001
```

Deixe este terminal aberto. Acesse `http://localhost:8001/docs` para testar a API.

### Passo 4: Configurar o Frontend PWA

Abra um **segundo terminal** na raiz do projeto:

```sh
cd desenvolvimento-apk

# Instalar dependências Node.js
npm install

# Configurar variáveis de ambiente
# Copie o .env.example e renomeie para .env
# Preencha com as chaves do Firebase/Supabase e a URL do backend:
# VITE_API_URL=http://<SEU_IP>:8001

# Iniciar o servidor de desenvolvimento
npm run dev
```

O frontend estará acessível em `http://localhost:3000`.

### Passo 5: Testar no Celular (Rede Local)

1. **Descubra o IP do PC:** Execute `ipconfig` (Windows) e anote o IPv4 (ex: `192.168.0.105`).

2. **Libere o Firewall (Windows):** Abra o PowerShell como Administrador:

```powershell
New-NetFirewallRule -DisplayName "Rayvora Dev 3000" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Rayvora Dev 8001" -Direction Inbound -LocalPort 8001 -Protocol TCP -Action Allow
```

3. **Acesse no Smartphone:** Abra o navegador e acesse `http://SEU_IP:3000`.

4. **Instale o App:** Toque no menu do navegador → **"Adicionar à Tela Inicial"**.

---

## Como Usar

1. **Login:** Acesse o app e entre com suas credenciais de operador.

2. **Nova Pesagem:**
   - Toque em **"Nova Avaliação"**.
   - Informe o identificador do brinco do animal (ex: `BOI_01`).
   - Tire uma foto ou faça upload de uma imagem do bovino.
   - O sistema retorna o peso estimado, a confiança e o **status de aptidão para abate**.

3. **Histórico:**
   - Visualize todas as pesagens registradas, sincronizadas em tempo real com o banco de dados.

4. **Análise de Evolução:**
   - Selecione um animal pelo brinco para visualizar o gráfico de evolução de peso e o Ganho Médio Diário (GMD).

---

## Modelos e Métricas

| Módulo | Técnica | Status | Resultado |
|---|---|---|---|
| Edge Impulse | Transfer Learning (MobileNet) | ✅ Produção | MAE: 32.87 kg |
| Fine-Tuning | Ajuste fino customizado (MobileNet) | 🧪 Experimental | MAE: 47.82 kg |
| Segmentação | YOLOv8s-Seg | ✅ Validado | mAP50: 0.9948 / mAP50-95: 0.9735 |

Detalhes completos de cada treinamento estão documentados nos READMEs de cada subpasta em `models/`.

---

## Roadmap (Próximos Passos)

- Reduzir o MAE do modelo de Fine-Tuning para superar o modelo atual em produção.
- Integrar o módulo de segmentação YOLOv8 ao pipeline do backend automaticamente.
- Publicar o backend de IA em servidor de nuvem (Railway ou Render) para produção sem rede local.
- Gerar o build de produção do PWA (`npm run build`) e hospedar o frontend.
- Expandir o dataset com mais imagens, raças e condições de captura variadas.

---

## Licença

Este projeto está licenciado sob a **MIT License**. Veja o arquivo `LICENSE` para mais detalhes.

---

## Autores

* **Ludivino José Da Silva**
* **Lucas Teixeira Belli**
* **Pedro Omna**

*Projeto Integrador I — Engenharia de Computação, UFSC (2026.1).*
