import streamlit as st
import os

# --- SUPRESSÃO DE LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURAÇÕES DE CAMINHO ---
# No Streamlit Cloud, BASE_DIR aponta para a raiz do repositório
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_SAVE_PATH = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

if not IMG_SAVE_PATH.exists():
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# --- DIAGNÓSTICO NO SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Diagnóstico Rayvora")
    
    # Verificação do Modelo
    if MODEL_PATH.exists():
        st.success("✅ Modelo H5: Localizado")
    else:
        st.error("❌ Modelo H5: Não encontrado")
    
    # Verificação e Inicialização do Banco
    def init_db():
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      brinco_id TEXT, 
                      data TEXT, 
                      peso REAL)''')
        conn.commit()
        conn.close()

    init_db()

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            count = pd.read_sql("SELECT COUNT(*) as n FROM pesagens", conn).iloc[0]['n']
            conn.close()
            st.metric("Total de Registros", f"{count} bois")
            if count == 0:
                st.warning("O histórico está vazio. Faça uma pesagem para começar.")
        except:
            st.error("Erro ao ler tabela de dados.")
    else:
        st.warning("Aguardando criação do banco...")

# --- CARREGAMENTO DE IA ---

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_weight_model():
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as e:
        st.error(f"Erro na IA de Peso: {e}")
        return None

# --- PROCESSAMENTO ---

def pipeline_visao(img_pil, _yolo):
    img_np = np.array(img_pil)
    results = _yolo(img_np, verbose=False)
    
    # Tenta recortar o boi
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 19: # Vaca/Boi
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                img_pil = img_pil.crop((x1, y1, x2, y2))
                break
                
    # Filtro CLAHE
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    res = clahe.apply(gray)
    final = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    return final

# --- UI PRINCIPAL ---

st.set_page_config(page_title="Rayvora Vision Pro", layout="wide", page_icon="🐂")
st.title("🐂 Rayvora Vision: Inteligência de Campo")

abas = st.tabs(["🚀 Nova Pesagem", "📊 Histórico e GMD", "⚙️ Configurações"])

with abas[0]:
    c1, c2 = st.columns(2)
    with c1:
        brinco = st.text_input("Identificação do Animal:", "BOI_")
        up = st.file_uploader("Foto para análise", type=['jpg', 'jpeg', 'png'])
        
    if up:
        img_raw = Image.open(up).convert('RGB')
        c1.image(img_raw, caption="Original", use_container_width=True)
        
        if st.button("🚀 Calcular Peso"):
            with st.spinner("Analisando morfologia..."):
                yolo = load_yolo()
                img_tratada = pipeline_visao(img_raw, yolo)
                c2.image(img_tratada, caption="Visão da IA", use_container_width=True)
                
                ia = load_weight_model()
                if ia:
                    prep = cv2.resize(img_tratada, (128, 128)) / 255.0
                    peso = float(ia.predict(np.expand_dims(prep, axis=0), verbose=False)[0][0])
                    
                    # Salvar dados
                    conn = sqlite3.connect(str(DB_PATH))
                    conn.execute("INSERT INTO pesagens (brinco_id, data, peso) VALUES (?,?,?)",
                                 (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso))
                    conn.commit()
                    conn.close()
                    
                    st.success(f"Peso Registrado: {peso:.2f} kg")
                    st.info("Os dados agora estão disponíveis na aba de Histórico.")

with abas[1]:
    st.subheader("Análise de Ganho Médio Diário (GMD)")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM pesagens", conn)
    conn.close()
    
    if not df.empty:
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        bois = df['brinco_id'].unique()
        sel = st.selectbox("Selecione o animal para o gráfico:", bois)
        
        df_filt = df[df['brinco_id'] == sel].sort_values('data')
        
        fig = px.line(df_filt, x='data', y='peso', markers=True, title=f"Evolução: {sel}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela simples
        st.dataframe(df_filt[['data', 'peso']], use_container_width=True)
    else:
        st.warning("Nenhum dado encontrado no banco de dados da nuvem.")

with abas[2]:
    st.subheader("Exportação de Dados")
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            st.download_button("📥 Baixar Banco de Dados (.db)", f, "rayvora_cloud.db")