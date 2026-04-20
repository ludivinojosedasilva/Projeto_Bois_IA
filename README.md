# 🐂 Rayvora - Monitoramento de Peso Bovino via IA

Este projeto utiliza **Visão Computacional** e **Deep Learning** para estimar o peso de gado de corte a partir de imagens da vista traseira (*back view*). Desenvolvido como parte do Projeto Integrador na **UFSC Araranguá**, a solução visa automatizar o monitoramento de peso, reduzindo o estresse do animal e otimizando a gestão da pecuária.

---

## 🚀 Funcionalidades
- **Estimativa de Peso:** Predição numérica (regressão) baseada em características morfológicas.
- **Data Augmentation:** Uso de `Albumentations` para aumentar a robustez do modelo contra variações de luz e ângulo.
- **Interface Web:** App interativo via `Streamlit` para produtores rurais.
- **Banco de Dados:** Persistência de dados em `SQLite` para acompanhamento do histórico de ganho de peso.
- **Treinamento em Nuvem:** Pipeline otimizado para Google Colab com uso de GPU.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.12 (Local) / 3.10 (Colab)
- **Framework IA:** TensorFlow 2.x e Keras.
- **Arquitetura:** MobileNetV2 com Transfer Learning.
- **Processamento:** OpenCV, Pandas, NumPy.
- **Dashboard:** Streamlit.

## 🧬 Metodologia e Arquitetura
O modelo foi construído utilizando a técnica de **Aprendizagem por Transferência (Transfer Learning)**. 
1. **Base:** MobileNetV2 (pré-treinada no ImageNet) para extração de características.
2. **Camadas Customizadas:** Global Average Pooling, Dropout (0.2) para evitar overfitting e uma camada Dense de saída para regressão.
3. **Otimização:** Erro Médio Absoluto (MAE) como função de perda para garantir precisão em quilogramas (kg).

## 📊 Resultados e Performance
O modelo foi validado com um dataset real de gado. 
- **Erro Médio Absoluto (MAE):** ~60.39 kg.
- **Comportamento:** O modelo demonstrou alta capacidade de identificar a tendência de peso, conforme validado pelo gráfico de dispersão (Real vs. Predito).

## 📁 Estrutura do Repositório
```text
├── src/                # Código fonte do Web App (app.py)
├── notebooks/          # Notebook (.ipynb) com o processo de treino
├── models/             # Arquivo do modelo treinado (.h5)
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação

📋 Como Rodar o Projeto
1. Clonar o repositório

git clone [https://github.com/SEU_USUARIO/Projeto_Bois_IA.git](https://github.com/SEU_USUARIO/Projeto_Bois_IA.git)
cd Projeto_Bois_IA
2. Configurar Ambiente Virtual

python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
3. Instalar Dependências

pip install -r requirements.txt
4. Executar o Aplicativo

streamlit run src/app.py
📧 Contato
Desenvolvido por Ludivino José Da Silva Estudante de Engenharia - UFSC Araranguá

Nota: Este é um protótipo acadêmico. Para uso comercial, recomenda-se o aumento do dataset e implementação de segmentação de imagem (YOLO).


---

### O que fazer agora:
1. **Crie o arquivo `requirements.txt`:** Na mesma pasta do README, coloque a lista de bibliotecas que te passei antes (tensorflow, streamlit, etc).
2. **Organize as pastas:** Coloque o seu `app.py` dentro de uma pasta chamada `src`.
3. **Commit e Push:** Envie tudo para o GitHub.