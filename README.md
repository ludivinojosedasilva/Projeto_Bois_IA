# Rayvora Vision Pro — Estimativa de Peso Bovino via Visão Computacional

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![TinyML: Edge Impulse](https://img.shields.io/badge/TinyML-Edge%20Impulse-1BA94C.svg)
![Status: Em Desenvolvimento](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)

**Autores:** Ludivino José Da Silva, Lucas Teixeira e Pedro Omna
**Disciplina:** Projeto Integrador I (2026.1) – Engenharia de Computação (UFSC)

---

## Sobre o Projeto

O **Rayvora Vision Pro** é um sistema inteligente de estimativa de peso de bovinos a partir de imagens, desenvolvido como parte do Projeto Integrador da UFSC. O objetivo é oferecer ao produtor rural uma alternativa de baixo custo e não invasiva à pesagem mecânica tradicional, usando apenas uma foto do animal e um modelo de **Machine Learning (TinyML)** para estimar o peso em quilogramas.

O sistema combina três frentes técnicas:

1. **Estimativa de peso (Edge Impulse / TinyML):** modelo de regressão treinado com Transfer Learning, atualmente em produção.
2. **Estimativa de peso (Fine-Tuning):** frente experimental, usando ajuste fino de um modelo pré-treinado (MobileNet) sobre dataset próprio.
3. **Segmentação de imagem (YOLOv8):** etapa de pré-processamento que isola o boi do fundo da imagem antes da estimativa de peso, aumentando a robustez do sistema.

Os resultados são exibidos em uma interface web (Streamlit), com histórico de pesagens armazenado em banco de dados local e gráficos de evolução de peso por animal.

---

## Arquitetura e Conceitos

O projeto segue um pipeline de **Visão Computacional aplicada à pecuária de precisão**, dividido em três módulos independentes:

1. **Módulo de Segmentação (YOLOv8)**
   Antes de qualquer estimativa de peso, uma rede de segmentação YOLOv8s-Seg identifica e isola o contorno do animal na imagem, removendo ruído de fundo (cercas, outros animais, vegetação). Isso padroniza a entrada para os modelos de regressão.

2. **Módulo de Estimativa via Edge Impulse (Produção)**
   Modelo de regressão treinado na plataforma Edge Impulse usando **Transfer Learning** sobre a arquitetura **MobileNet**. É o modelo atualmente utilizado pela aplicação web. Resultado atual: **MAE de 32.87 kg**.

3. **Módulo de Estimativa via Fine-Tuning (Experimental)**
   Frente de pesquisa paralela, em que um modelo MobileNet pré-treinado (obtido de um projeto de referência de outro TCC) passa por **Fine-Tuning** com o dataset próprio da equipe, buscando reduzir ainda mais o erro de estimativa. Resultado atual: **MAE de 47.82 kg** (ainda não supera o modelo de produção).

Cada módulo possui documentação técnica detalhada na sua respectiva pasta (ver seção [Estrutura do Repositório](#estrutura-do-repositório)).

---

## Funcionalidades

- **Upload e Estimativa de Peso:** o usuário envia uma foto do animal e o sistema retorna o peso estimado, junto com uma métrica de confiança e margem de erro.
- **Inferência com Quantificação de Incerteza:** múltiplas inferências com ruído estocástico são realizadas para calcular a confiança da predição e a margem de erro estatística.
- **Histórico Persistente:** todas as pesagens são salvas em um banco de dados local (SQLite), incluindo brinco do animal, data, peso estimado, confiança, erro e a foto utilizada.
- **Análise de Evolução:** seleção de um animal já cadastrado para visualizar a curva de ganho de peso ao longo do tempo, com gráfico de linha e estatísticas (peso mínimo, máximo e atual).
- **Motor Portátil (LiteRT):** inferência executada via `ai-edge-litert`, leve e compatível com ambientes de nuvem com recursos limitados (como o Streamlit Community Cloud).

---

## Requisitos de Hardware e Software

### Hardware
- Computador para desenvolvimento e treinamento (testado em Windows 11).
- Smartphone ou câmera para captura das imagens dos bovinos.
- (Opcional, para etapas futuras de TinyML embarcado) Microcontrolador compatível com TensorFlow Lite for Microcontrollers.

### Software
- **Linguagem:** Python 3.12+
- **IDE recomendada:** VS Code (com terminal integrado) ou qualquer editor de preferência.
- **Plataforma de treinamento:** [Edge Impulse](https://edgeimpulse.com/) (conta gratuita) e Google Colab (para o módulo de Fine-Tuning).
- **Gerenciador de pacotes:** `pip`.
- **Controle de versão:** `git`.

### Bibliotecas Python (`requirements.txt`)

| Biblioteca | Função no projeto |
|---|---|
| `streamlit` | Framework da interface web (upload de imagem, dashboard, histórico) |
| `ai-edge-litert` | Motor de inferência portátil (LiteRT/TFLite) usado para carregar e executar o modelo `.tflite` do Edge Impulse |
| `opencv-python-headless` | Processamento digital de imagens (redimensionamento, conversão de espaço de cor) |
| `numpy` | Operações numéricas e manipulação de arrays (normalização de pixels, cálculo de estatísticas) |
| `pillow` | Abertura, conversão e manipulação de imagens enviadas pelo usuário |
| `pandas` | Manipulação de dados tabulares (leitura e exibição do histórico de pesagens) |
| `matplotlib` | Geração de gráficos auxiliares (quando necessário fora do `st.line_chart` nativo) |

Instalação:
```sh
pip install -r requirements.txt
```

> **Observação:** o módulo de segmentação (YOLOv8) e o notebook de Fine-Tuning utilizam bibliotecas adicionais (`ultralytics`, `torch`, etc.) que **não** fazem parte do `requirements.txt` principal, pois não são necessárias para rodar a aplicação web em produção — são usadas apenas durante o desenvolvimento/treinamento dos modelos (ver READMEs específicos em `models/segmentation/` e `models/fine-tuning/`).

---

## Estrutura do Repositório
Projeto_Bois_IA/

├── src/

│   ├── app.py              # Aplicação principal (Streamlit) — ponto de entrada em produção

│   └── main.py              # Script auxiliar (em avaliação)

│

├── models/

│   ├── segmentation/         # Modelo YOLOv8 de segmentação do boi na imagem

│   ├── edge-impulse/         # Modelo de regressão em PRODUÇÃO (Transfer Learning, MAE 32.87 kg)

│   └── fine-tuning/          # Modelo experimental via Fine-Tuning (MAE 47.82 kg)

│

├── data/

│   └── dataset/              # Dataset de imagens + pesos reais (fora do Git, ver README da pasta)

│

├── external/

│   └── TCC_V2-master.zip     # Projeto de referência (TCC de outro grupo) usado como base do Fine-Tuning

│

├── requirements.txt

├── .gitignore

└── README.md

Cada subpasta de `models/` possui seu próprio `README.md` com o passo a passo técnico completo de como aquele modelo específico foi construído, as métricas obtidas e o estado atual (produção ou experimental).

---

## Instalação e Configuração

### Passo 1: Clonar o repositório

```sh
git clone https://github.com/ludivinojosedasilva/Projeto_Bois_IA.git
cd Projeto_Bois_IA
```

### Passo 2: Instalar as dependências

```sh
pip install -r requirements.txt
```

### Passo 3: Executar a aplicação

```sh
streamlit run src/app.py
```

A aplicação abrirá automaticamente no navegador, em `http://localhost:8501`.

---

## Como Usar

1. **Nova Pesagem**
   - No menu lateral, selecione **"Nova Pesagem"**.
   - Informe o identificador do brinco do animal (ex: `BOI_01`).
   - Faça o upload de uma foto do bovino.
   - Clique em **"Calcular Peso"**.
   - O sistema exibirá o peso estimado, a confiança da predição e a margem de erro.

2. **Histórico**
   - No menu lateral, selecione a aba de histórico para visualizar todas as pesagens já registradas.

3. **Análise de Evolução**
   - Selecione o brinco de um animal já pesado anteriormente.
   - Visualize o gráfico de evolução de peso e as estatísticas (peso mínimo, máximo e mais recente).

---

## Modelos e Métricas

| Módulo | Técnica | Status | MAE |
|---|---|---|---|
| Edge Impulse | Transfer Learning (MobileNet) | ✅ Produção | 32.87 kg |
| Fine-Tuning | Fine-Tuning sobre modelo pré-treinado (MobileNet) | 🧪 Experimental | 47.82 kg |
| Segmentação | YOLOv8s-Seg | ✅ Validado | mAP50: 0.9948 / mAP50-95: 0.9735 |

Detalhes completos de cada treinamento (hiperparâmetros, datasets, passo a passo) estão documentados nos READMEs de cada subpasta em `models/`.

---

## Roadmap (Próximos Passos)

- Reduzir o MAE do modelo de Fine-Tuning para que supere o modelo atual em produção.
- Integrar a etapa de segmentação (YOLOv8) ao pipeline da aplicação web, isolando o boi do fundo antes da estimativa de peso.
- Avaliar viabilidade de embarcar o pipeline completo (segmentação + regressão) em ambientes com recursos limitados.
- Expandir o dataset com mais imagens e condições de captura (iluminação, ângulo, raça).

---

## Licença

Este projeto está licenciado sob a **MIT License**. Veja o arquivo `LICENSE` para mais detalhes.

---

## Autores

* **Ludivino José Da Silva**
* **Lucas Teixeira**
* **Pedro Omna**

*Projeto Integrador I — Engenharia de Computação, UFSC (2026.1).*
