# Edge Impulse — Modelo de Estimativa de Peso (Produção)

> **Status: ✅ Modelo em produção.** Este é o modelo atualmente usado pelo sistema (`src/app.py`), carregado a partir de `models/edge-impulse/model.tflite`.
> Membros da equipe responsável pelo desenvolvimento de Machine Learning desenvolveram modelos de Regressão/Transfer Learning com hiperparâmetros e atributos de imagem diferentes, a título de comparação e busca do modelo que mais desempenha melhor para resultados satisfatórios e mais estáveis de predição de peso.

## Resumo

Modelo de regressão para estimativa de peso de bovinos a partir de imagens, treinado na plataforma [Edge Impulse](https://edgeimpulse.com/) usando **Transfer Learning** com base na arquitetura **MobileNet**. 

## Métricas obtidas

| Métrica | Valor |
|---|---|
| MAE (Mean Absolute Error) | **32.87 kg** |
| MSE (Mean Squared Error) | 1539.54 |
| RMSE (raiz do MSE) | ≈ 39.23 kg |
| Explained Variance Score | -1.19 |

**Interpretação:**
- Na média, o modelo erra a estimativa de peso em ±32.87 kg.
- O RMSE maior que o MAE indica que existem alguns casos (outliers) onde o erro é bem maior que a média — provavelmente fotos com ângulo, iluminação ou enquadramento ruins.
- O Explained Variance Score negativo indica que o modelo ainda não está capturando bem a variância real dos pesos — há espaço de melhoria, e por isso a equipe está testando a abordagem de Fine-Tuning (ver `models/fine-tuning/`) como próximo passo.

## Passo a passo: como o modelo foi construído

### 1. Aquisição e upload do dataset
- Imagens de bovinos com os respectivos pesos reais (kg) como label, vindas do processamento feito no Google Colab.
- Upload feito em **Data Acquisition**, com divisão automática de **80% treino / 20% teste**.

### 2. Configuração do Impulse (Impulse Design)
- **Bloco de entrada (Image Data):** redimensionamento para **128×128 pixels**, modo *Resize: Squash*.
- **Bloco de processamento (Processing Block):** bloco **Image**, extraindo características em **RGB**.
- **Bloco de aprendizado (Learning Block):** **Transfer Learning (Images)**, usando a arquitetura **MobileNet** pré-treinada como base.

### 3. Extração de características
- Em **Image**, parâmetros salvos com normalização padrão de pixels.
- **Generate Features** executado — o Edge Impulse processa todas as imagens e gera o **Feature Explorer** (gráfico 3D mostrando o agrupamento visual das imagens conforme similaridade).

### 4. Treinamento (Transfer Learning)
- Base **MobileNet** com pesos pré-treinados, ajustando as camadas finais para a tarefa de regressão (peso em kg).
- Ajuste de hiperparâmetros (épocas e taxa de aprendizado) para buscar convergência estável.

### 5. Validação
- Métricas exibidas no painel após o treino (ver tabela acima).
- Gráfico **Predicted vs. Actual**: dispersão comparando peso real (eixo X) vs. peso previsto (eixo Y).

### 6. Exportação (Deployment)
- Aba **Deployment** → formato **C++ Library**.
- Otimização: **Float32** (sem quantização), para preservar a precisão decimal necessária à regressão.
- **Build** gerado, baixando o `.zip` com o `model.tflite` correspondente.

## Arquivos nesta pasta

| Arquivo | Descrição |
|---|---|
| `model.tflite` | Modelo final extraído, usado em produção por `src/app.py` |
| `edge-impulse-export.zip` | Pacote completo de export baixado do Edge Impulse (contém o `.tflite` + metadados/código C++) |

## Dataset utilizado

Ver `data/dataset/README.md` — mesmo dataset usado também no Fine-Tuning.

## Glossário rápido (termos do Edge Impulse)

| Termo (EN) | Significado (PT) |
|---|---|
| Data Acquisition | Aquisição/coleta de dados |
| Impulse Design | Desenho do fluxo de processamento (do dado bruto até a saída) |
| Features | Características extraídas da imagem (bordas, texturas, contornos) |
| Transfer Learning | Reaproveitar uma rede já treinada (MobileNet) e ajustar só a parte final para a tarefa específica |
| Epochs | Épocas — número de vezes que a rede percorre todo o dataset de treino |
| Learning Rate | Taxa de aprendizado — tamanho do "passo" de ajuste dos pesos a cada erro |
| MAE | Erro médio absoluto — erro médio em kg |
| Explained Variance Score | O quanto o modelo explica a variação real dos pesos (ideal: próximo de 1.0) |
| Deployment | Exportação do modelo treinado para uso fora da plataforma |

## Autores

Ludivino José da Silva, Lucas Teixeira Belli
