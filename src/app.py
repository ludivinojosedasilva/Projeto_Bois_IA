import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import os

# --- CONFIGURAÇÕES DE CAMINHO (Versão Cloud/GitHub) ---
# Em vez de usar o Drive, usamos pastas locais no repositório
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'monitoramento_bois.db')
IMG_SAVE_PATH = os.path.join(BASE_DIR, '..', 'fotos_pesagens')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_peso_bois.h5')

# Criar pasta para fotos se não existir
if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  brinco_id TEXT, 
                  data TEXT, 
                  peso_estimado REAL,
                  caminho_foto TEXT)''')
    conn.commit()
    conn.close()

@st.cache_resource
def load_model_ia():
    # Carrega o modelo de forma resiliente
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mae')
    return model

def verificar_se_e_boi(img_input, modelo):
    """Filtro básico de segurança para validar a entrada"""
    pred = modelo.predict(img_input)
    peso = float(pred[0][0])
    # Se o peso for absurdo para um bovino, recusamos a imagem
    if peso < 50 or peso > 1500:
        return False, peso
    return True, peso

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Projeto Integrador - UFSC", layout="wide")
init_db()

st.title("🐂 Monitoramento de Peso Bovino")
st.markdown("Protótipo de Pesagem Visual para a disciplina de Projeto Integrador")

menu = ["Nova Pesagem", "Histórico e Auditoria"]
escolha = st.sidebar.selectbox("Menu", menu)

if escolha == "Nova Pesagem":
    st.header("⚖️ Realizar Pesagem")
    brinco = st.text_input("Identificação do Animal (Brinco):", "ID_")
    foto = st.file_uploader("Carregar foto da traseira", type=['jpg', 'jpeg', 'png'])
    
    if foto is not None:
        img_original = Image.open(foto).convert('RGB')
        st.image(img_original, caption="Imagem para análise", width=400)
        
        if st.button("🚀 Calcular Peso"):
            # Carregar modelo apenas no clique para economizar memória
            model = load_model_ia()
            
            # Pré-processamento
            img_arr = np.array(img_original)
            img_res = cv2.resize(img_arr, (128, 128)) / 255.0
            img_input = np.expand_dims(img_res, axis=0)
            
            # Validação
            e_boi, peso_final = verificar_se_e_boi(img_input, model)
            
            if not e_boi:
                st.error("❌ Imagem Inválida: O sistema não detetou características de um bovino adulto.")
            else:
                # Salvar Foto
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_foto = f"{brinco}_{timestamp}.jpg"
                caminho_completo = os.path.join(IMG_SAVE_PATH, nome_foto)
                img_original.save(caminho_completo)
                
                # Salvar no Banco
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                c.execute("INSERT INTO pesagens (brinco_id, data, peso_estimado, caminho_foto) VALUES (?, ?, ?, ?)", 
                          (brinco, agora, peso_final, nome_foto))
                conn.commit()
                conn.close()
                
                st.success(f"✅ Pesagem Registada: {peso_final:.2f} kg")

elif escolha == "Histórico e Auditoria":
    st.header("📈 Histórico de Pesagens")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM pesagens", conn)
        conn.close()
        
        if not df.empty:
            st.dataframe(df[['brinco_id', 'data', 'peso_estimado']], use_container_width=True)
            
            st.divider()
            st.subheader("🔍 Ver Foto da Pesagem")
            id_busca = st.selectbox("Selecione o ID para auditoria:", df['id'])
            nome_arq = df[df['id'] == id_busca]['caminho_foto'].values[0]
            caminho_arq = os.path.join(IMG_SAVE_PATH, nome_arq)
            
            if os.path.exists(caminho_arq):
                st.image(Image.open(caminho_arq), width=500)
    else:
        st.info("Ainda não existem pesagens registadas.")