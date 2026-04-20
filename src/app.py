import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Rayvora - Monitoramento de Gado", layout="wide")

# --- BANCO DE DADOS ---
def init_db():
    conn = sqlite3.connect('monitoramento_bois.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  brinco_id TEXT, 
                  data TEXT, 
                  peso_estimado REAL)''')
    conn.commit()
    conn.close()

# --- CARREGAR MODELO ---
@st.cache_resource
def load_model():
    # Certifique-se de que o arquivo .h5 está na mesma pasta do app.py
    return tf.keras.models.load_model('modelo_peso_bois.h5')

model = load_model()
init_db()

# --- INTERFACE ---
st.title("📊 Sistema de Monitoramento de Peso - IA")
st.sidebar.header("Menu de Navegação")
opcao = st.sidebar.selectbox("Escolha uma opção", ["Nova Pesagem", "Histórico e Análise"])

if opcao == "Nova Pesagem":
    st.header("⚖️ Estimar Peso por Imagem")
    
    brinco = st.text_input("ID ou Brinco do Animal:", "Ex: Boi_001")
    upload = st.file_uploader("Carregar foto da traseira (Back View)", type=['png', 'jpg', 'jpeg'])

    if upload is not None:
        image = Image.open(upload)
        st.image(image, caption='Imagem Carregada', width=300)
        
        if st.button("Calcular Peso"):
            # Pré-processamento igual ao que fizemos no Colab
            img_array = np.array(image.convert('RGB'))
            img_resized = cv2.resize(img_array, (128, 128))
            img_normalized = img_resized / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)
            
            # Predição
            predicao = model.predict(img_input)
            peso_final = float(predicao[0][0])
            
            st.success(f"Peso Estimado: **{peso_final:.2f} kg**")
            
            # Salvar no Banco de Dados
            conn = sqlite3.connect('monitoramento_bois.db')
            c = conn.cursor()
            data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO pesagens (brinco_id, data, peso_estimado) VALUES (?, ?, ?)", 
                      (brinco, data_atual, peso_final))
            conn.commit()
            conn.close()
            st.info("Dados salvos no histórico com sucesso!")

elif opcao == "Histórico e Análise":
    st.header("📈 Histórico de Monitoramento")
    
    conn = sqlite3.connect('monitoramento_bois.db')
    df = pd.read_sql_query("SELECT * FROM pesagens ORDER BY data DESC", conn)
    conn.close()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # Gráfico de evolução de um animal específico
        st.subheader("Evolução por Animal")
        animal_selecionado = st.selectbox("Selecione o Brinco:", df['brinco_id'].unique())
        df_animal = df[df['brinco_id'] == animal_selecionado].sort_values('data')
        
        st.line_chart(data=df_animal, x='data', y='peso_estimado')
    else:
        st.warning("Nenhum dado registrado ainda.")