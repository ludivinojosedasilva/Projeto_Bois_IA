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
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_SAVE_PATH = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

if not IMG_SAVE_PATH.exists():
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# --- BANCO DE DADOS COM MIGRAÇÃO AUTOMÁTICA ---
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # 1. Cria a tabela básica se não existir
    cursor.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      brinco_id TEXT, 
                      data TEXT, 
                      peso REAL)''')
    
    # 2. MÁGICA DA MIGRAÇÃO: Verifica se a coluna 'foto_nome' existe
    cursor.execute("PRAGMA table_info(pesagens)")
    colunas = [col[1] for col in cursor.fetchall()]
    
    if 'foto_nome' not in colunas:
        st.toast("Atualizando estrutura do banco de dados...", icon="🔧")
        cursor.execute("ALTER TABLE pesagens ADD COLUMN foto_nome TEXT DEFAULT 'sem_foto.jpg'")
    
    conn.commit()
    conn.close()

# --- CARREGAMENTO DE MODELOS ---
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    ia_peso = None
    if MODEL_PATH.exists():
        try:
            ia_peso = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        except Exception as e:
            st.error(f"Erro ao carregar modelo H5: {e}")
    return yolo, ia_peso

# --- PROCESSAMENTO VISUAL ---
def pipeline_visao(img_pil, _yolo):
    img_np = np.array(img_pil)
    results = _yolo(img_np, verbose=False)
    
    detectado = False
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 19 and float(box.conf) > 0.6: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                img_pil = img_pil.crop((x1, y1, x2, y2))
                detectado = True
                break
                
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    res = clahe.apply(gray)
    final = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    return final, detectado

# --- UI PRINCIPAL ---
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide", page_icon="🐂")
init_db()
yolo, ia_peso = load_models()

st.title("🐂 Rayvora Vision Pro: Inteligência de Campo")

abas = st.tabs(["🚀 Nova Pesagem", "📊 Histórico e GMD", "📦 Data Hub (Retreino)"])

with abas[0]:
    c1, c2 = st.columns(2)
    with c1:
        brinco = st.text_input("Identificação do Animal:", "BOI_")
        up = st.file_uploader("Foto para análise", type=['jpg', 'jpeg', 'png'])
        
    if up:
        img_raw = Image.open(up).convert('RGB')
        c1.image(img_raw, caption="Original", width=400)
        
        if st.button("🚀 Calcular Peso"):
            with st.spinner("Analisando morfologia..."):
                img_tratada, detectou = pipeline_visao(img_raw, yolo)
                
                if not detectou:
                    st.error("❌ Nenhum bovino detectado com confiança > 60%.")
                else:
                    c2.image(img_tratada, caption="Visão da IA", width=400)
                    
                    if ia_peso:
                        prep = cv2.resize(img_tratada, (128, 128)) / 255.0
                        peso = float(ia_peso.predict(np.expand_dims(prep, axis=0), verbose=False)[0][0])
                        
                        # Salvar com nome de arquivo único
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_f = f"{brinco}_{ts}.jpg"
                        img_raw.save(IMG_SAVE_PATH / nome_f)
                        
                        conn = sqlite3.connect(str(DB_PATH))
                        conn.execute("INSERT INTO pesagens (brinco_id, data, peso, foto_nome) VALUES (?,?,?,?)",
                                     (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso, nome_f))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"✅ Peso Registrado: {peso:.2f} kg")

with abas[1]:
    st.subheader("Análise de Desempenho")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM pesagens", conn)
    conn.close()
    
    if not df.empty:
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        sel = st.selectbox("Selecione o animal:", df['brinco_id'].unique())
        df_filt = df[df['brinco_id'] == sel].sort_values('data')
        
        fig = px.line(df_filt, x='data', y='peso', markers=True, title=f"Evolução: {sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhum dado encontrado.")

with abas[2]:
    st.subheader("📦 Galeria de Refinamento")
    conn = sqlite3.connect(str(DB_PATH))
    # Pegamos os últimos 12 para a galeria
    registros = pd.read_sql("SELECT * FROM pesagens ORDER BY id DESC LIMIT 12", conn)
    conn.close()
    
    if not registros.empty:
        # Criar grid de fotos
        for i in range(0, len(registros), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(registros):
                    row = registros.iloc[i + j]
                    caminho = IMG_SAVE_PATH / str(row['foto_nome'])
                    if caminho.exists():
                        cols[j].image(str(caminho), caption=f"{row['brinco_id']}\n{row['peso']:.1f}kg")
    else:
        st.info("A galeria aparecerá após as primeiras pesagens.")