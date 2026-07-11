# Modelo de Segmentação — YOLOv8s-Seg

**Status:** ✅ Modelo treinado e validado. Integrado ao pipeline da aplicação via `src/app.py` e `src/main.py`.

---

## Sobre

Este módulo implementa a etapa de **análise de qualidade de imagem** do sistema Rayvora Vision Pro. Antes de qualquer estimativa de peso, um modelo de segmentação de instância baseado em YOLOv8s-Seg identifica e isola o contorno do animal na imagem, extraindo métricas morfológicas (área da máscara, aspect ratio do bounding box, brilho médio dentro da região segmentada) que são usadas para gerar avisos ao usuário sobre a qualidade da foto capturada.

O modelo **não substitui** o modelo de predição de peso (Edge Impulse) — ele atua como uma etapa anterior, gerando warnings quando a imagem apresenta problemas como iluminação insuficiente, animal muito longe da câmera ou posição lateral incorreta. Todos os avisos são informativos: a estimativa de peso é realizada independentemente (exceto no que se refere ao check de validação de animal na imagem, que autoriza ou rejeita a imagem para o prosseguimento do fluxo da aplicação): 

---

## Arquitetura

| Item | Detalhe |
|------|---------|
| Arquitetura | YOLOv8s-Seg (small, segmentação de instância) |
| Framework | PyTorch via Ultralytics 8.4.71 |
| Parâmetros | ~11,79 milhões |
| Tamanho do modelo | ~22 MB (.pt) |
| Tarefa | Segmentação de instância (1 classe: `cow`) |

---

## Dataset

- **Origem:** Dataset público do Kaggle, com imagens de bovinos anotadas no formato YOLO
- **Total de imagens válidas após limpeza:** ~6.009 (treino) / 1.503 (validação)
- **Divisão:** 80% treino / 20% validação
- **Formato das anotações:** YOLO segmentação (.txt com polígonos normalizados)
- **Resolução de entrada:** 640×640 px (redimensionamento automático)

### Limpeza realizada antes do treino

O dataset original apresentava dois problemas que foram corrigidos:

1. **Duas classes:** o dataset original tinha classe `0` (cow) e classe `1` (sticker — marcas d'água de watermark). Todas as anotações de classe `1` foram convertidas para classe `0`, unificando o dataset em uma única classe.
2. **Polígonos degenerados:** anotações com menos de 6 pontos ou área de bounding box inferior a 0,1% da imagem foram removidas automaticamente, pois representavam ruído de anotação e causavam rejeição de arquivos inteiros pelo Ultralytics.

---

## Treinamento

Realizado no Google Colab com GPU Tesla T4.

| Parâmetro | Valor |
|-----------|-------|
| `epochs` | 100 (early stopping em ~57) |
| `imgsz` | 640 |
| `batch` | 16 |
| `optimizer` | AdamW |
| `lr0` | 0.001 |
| `weight_decay` | 0.0005 |
| `patience` | 20 (early stopping) |
| `augment` | True |
| `cache` | False |
| `save_period` | 10 (checkpoint a cada 10 épocas) |
| Pesos iniciais | `yolov8s-seg.pt` (pré-treinado ImageNet) |

O treino foi encerrado automaticamente pelo early stopping na época ~57, quando o `mAP50-95` da máscara estabilizou sem melhora por 20 épocas consecutivas.

---

## Resultados

Métricas avaliadas no conjunto de validação (1.503 imagens):

| Métrica | Valor |
|---------|-------|
| mAP50 (máscara) | **0.9948** |
| mAP50-95 (máscara) | **0.9735** |
| mAP50 (bounding box) | 0.9948 |
| mAP50-95 (bounding box) | 0.9899 |
| Precisão (máscara) | ~0.998 |
| Recall (máscara) | ~0.998 |

> **mAP50** mede a precisão considerando correto qualquer predição com sobreposição ≥ 50% em relação à anotação real.
> **mAP50-95** é a média do mAP calculado de 50% a 95% de sobreposição — critério mais rigoroso que avalia a precisão do contorno nos detalhes.

---

## Sistema de Warnings (análise de qualidade)

A partir da máscara gerada pelo modelo, são extraídas métricas morfológicas que alimentam 4 checks de qualidade. Os checks geram avisos informativos ao usuário — **nunca bloqueiam a estimativa de peso**.

| Check | Feature extraída | Threshold | Warning gerado |
|-------|-----------------|-----------|----------------|
| Detecção | `confidence` (YOLO) | < 0.50 | "Nenhum animal detectado" |
| Iluminação | `brightness_mean` (brilho médio dentro da máscara) | < 30.0 | "Iluminação insuficiente" |
| Distância | `mask_area_ratio` (% da imagem ocupada pela máscara) | < 5% | "Animal muito longe da câmera" |
| Ângulo | `bbox_aspect_ratio` (largura/altura do bounding box) | > 1.30 | "Animal possivelmente de lado" |

### Observações sobre calibração

Os thresholds acima foram definidos após testes com imagens reais do projeto:

- **Brilho (`brightness_min = 30.0`):** o valor inicial de 50.0 gerava warnings em imagens com iluminação adequada. O valor 30.0 mostrou-se mais permissivo e adequado às condições reais de captura em campo.
- **Aspect ratio (`aspect_ratio_max = 1.30`):** testes mostraram que bois fotografados de lado apresentavam ratio em torno de 1.3, valor inferior ao esperado teoricamente (~1.7-2.0). Isso ocorre porque o ângulo da câmera e a distância de captura reduzem a diferença geométrica entre pose lateral e rear view. O threshold 1.30 foi o que melhor separou os dois casos nas imagens testadas. Recomenda-se nova calibração com dataset maior usando os nomes de arquivo como referência (sufixo `_r_` = rear, `_s_` = side).
- **Distância (`mask_area_ratio_min = 0.05`):** em testes com imagens de bois muito distantes, o próprio modelo de segmentação não detectava o animal com confiança suficiente, o que já ativava o check de detecção. O threshold de 5% mostrou-se adequado como segunda camada de verificação.

---

## Arquivos nesta pasta

| Arquivo | Descrição |
|---------|-----------|
| `best.pt` | Modelo final (melhor época) — usado em produção |
| `modelo_bois.onnx` | Export ONNX — multiplataforma, ideal para API |
| `modelo_bois.tflite` | Export TFLite float32 — para ambientes com recursos limitados |
| `segmentacao_bois_yolov8.ipynb` | Notebook completo de treino, avaliação e testes |
| `MODELOS_README.md` | Descrição resumida dos formatos exportados |

---

## Como usar o modelo

```python
from ultralytics import YOLO

model = YOLO('models/segmentation/best.pt')
results = model.predict('imagem.jpg', conf=0.5)

# Acessa a máscara da melhor detecção
result = results[0]
if result.masks is not None:
    mask = result.masks.data[0].cpu().numpy()
```

Para o pipeline completo de extração de features e geração de warnings, ver `src/app.py` (Streamlit) ou `src/main.py` (FastAPI/Node).

---

## Dependências

```
ultralytics==8.4.71
opencv-python-headless
numpy
pillow
```

Estão disponíveis três formatos exportados — modelo_bois.pt (PyTorch, formato nativo, recomendado para uso com a biblioteca Ultralytics), modelo_bois.onnx (formato leve e multiplataforma) e modelo_bois.tflite (otimizado para dispositivos com recursos limitados, disponível em precisão float32 e float16).

## Autor

Lucas Teixeira Belli
