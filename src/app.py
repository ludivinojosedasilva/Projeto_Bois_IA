import streamlit as st
import os

# --- SUPRESSÃO DE LOGS (Deve vir antes de tudo) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencia avisos do TensorFlow

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

# --- CONFIGURAÇÕES DE CAMINHO ROBUSTAS ---
# Descobre onde o app.py está rodando
BASE_DIR = Path(__file__).resolve().parent.parent # Sobe um nível para a raiz do projeto

# Define caminhos absolutos
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_SAVE_PATH = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

# Garante a existência da pasta de fotos
if not IMG_SAVE_PATH.exists():
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# --- FUNÇÕES DE DIAGNÓSTICO NO SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Painel de Controle Rayvora")
    st.info(f"Ambiente: {BASE_DIR}")
    
    if MODEL_PATH.exists():
        st.success("✅ Modelo .h5 localizado")
    else:
        st.error(f"❌ Modelo .h5 não encontrado em: {MODEL_PATH}")
        # Busca alternativa
        alt_path = Path(__file__).resolve().parent / "models" / "modelo_peso_bois.h5"
        if alt_path.exists():
            MODEL_PATH = alt_path
            st.warning("⚠️ Modelo achado em caminho alternativo.")

# --- CARREGAMENTO DE MODELOS (CACHE) ---

@st.cache_resource
def load_yolo():
    # O YOLOv8n será baixado automaticamente na primeira execução
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_weight_model():
    try:
        # Carregamento 'Lazy' para economizar memória
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar IA de Peso: {e}")
        return None

# --- PROCESSAMENTO VISUAL ---

def detectar_boi(img_pil, _yolo):
    img_np = np.array(img_pil)
    results = _yolo(img_np, verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 19: # 19 = cow no COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Crop com margem
                return img_pil.crop((x1, y1, x2, y2)), True
    return img_pil, False

def tratar_clahe(img_pil):
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    res = clahe.apply(gray)
    return cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

# --- BANCO DE DADOS ---

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, brinco_id TEXT, data TEXT, peso REAL)''')
    conn.commit()
    conn.close()

def get_gmd(brinco, peso_atual):
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(f"SELECT data, peso FROM pesagens WHERE brinco_id='{brinco}' ORDER BY id DESC LIMIT 1", conn)
    conn.close()
    if not df.empty:
        p_ant = df['peso'].iloc[0]
        d_ant = datetime.strptime(df['data'].iloc[0], "%d/%m/%Y %H:%M:%S")
        dias = (datetime.now() - d_ant).days or 1
        return (peso_atual - p_ant)/dias, dias, p_ant
    return None, None, None

# --- UI PRINCIPAL ---

st.set_page_config(page_title="Rayvora Vision Pro", layout="wide")
init_db()

st.title("🐂 Rayvora Vision: Gestão de Rebanho")

abas = st.tabs(["🚀 Nova Pesagem", "📊 Analytics", "📁 Arquivos"])

with abas[0]:
    c1, c2 = st.columns(2)
    with c1:
        brinco = st.text_input("ID do Brinco:", "BOI_")
        up = st.file_uploader("Foto do Animal", type=['jpg', 'png', 'jpeg'])
    
    if up:
        img = Image.open(up).convert('RGB')
        c1.image(img, caption="Original", use_container_width=True)
        
        if st.button("Executar Pesagem"):
            with st.spinner("IA Processando..."):
                # YOLO
                yolo = load_yolo()
                recorte, achei = detectar_boi(img, yolo)
                
                # CLAHE
                final_viz = tratar_clahe(recorte)
                c2.image(final_viz, caption="Filtro Morfológico", use_container_width=True)
                
                # PESO
                ia = load_weight_model()
                if ia:
                    prep = cv2.resize(final_viz, (128, 128)) / 255.0
                    peso = float(ia.predict(np.expand_dims(prep, axis=0), verbose=False)[0][0])
                    
                    if 30 < peso < 1800:
                        gmd, dias, p_ant = get_gmd(brinco, peso)
                        
                        # Salvar
                        conn = sqlite3.connect(str(DB_PATH))
                        conn.execute("INSERT INTO pesagens (brinco_id, data, peso) VALUES (?,?,?)",
                                     (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"Peso Estimado: {peso:.2f} kg")
                        if gmd is not None:
                            st.metric("GMD", f"{gmd:.3f} kg/dia", delta=f"{peso-p_ant:.2f} kg")
                    else:
                        st.error("Erro de leitura: Animal fora dos padrões de peso.")

with abas[1]:
    st.subheader("Histórico de Crescimento")
    conn = sqlite3.connect(str(DB_PATH))
    df_all = pd.read_sql("SELECT * FROM pesagens", conn)
    conn.close()
    
    if not df_all.empty:
        df_all['data'] = pd.to_datetime(df_all['data'], dayfirst=True)
        sel = st.selectbox("Escolha o animal:", df_all['brinco_id'].unique())
        df_filt = df_all[df_all['brinco_id'] == sel].sort_values('data')
        fig = px.line(df_filt, x='data', y='peso', markers=True, title=f"Curva de Peso: {sel}")
        st.plotly_chart(fig, use_container_width=True)

with abas[2]:
    st.subheader("Gerenciamento de Dados")
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            st.download_button("📥 Baixar Banco de Dados (.db)", f, "rayvora_data.db")