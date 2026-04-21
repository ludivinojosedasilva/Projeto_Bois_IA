import streamlit as st
import os

# --- SUPRESSÃO DE LOGS DO TENSORFLOW ---
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
from pathlib import Path  # Correção: pathlib minúsculo

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_SAVE_PATH = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

# Garante que a pasta de fotos para retreino exista
if not IMG_SAVE_PATH.exists():
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# --- BANCO DE DADOS COM MIGRAÇÃO AUTOMÁTICA ---
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # Cria tabela se não existir
    cursor.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      brinco_id TEXT, 
                      data TEXT, 
                      peso REAL)''')
    
    # Verifica se a coluna 'foto_nome' existe (Migração)
    cursor.execute("PRAGMA table_info(pesagens)")
    colunas = [col[1] for col in cursor.fetchall()]
    if 'foto_nome' not in colunas:
        cursor.execute("ALTER TABLE pesagens ADD COLUMN foto_nome TEXT DEFAULT 'sem_foto.jpg'")
    
    conn.commit()
    conn.close()

# --- CARREGAMENTO DE MODELOS (CACHE) ---
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    ia_peso = None
    if MODEL_PATH.exists():
        try:
            ia_peso = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        except Exception as e:
            st.error(f"Erro ao carregar modelo de peso: {e}")
    return yolo, ia_peso

# --- PIPELINE DE VISÃO CALIBRADO ---
def pipeline_visao(img_pil, _yolo):
    img_np = np.array(img_pil)
    # Calibração: Confiança inicial de 0.25 para não ignorar o boi
    results = _yolo(img_np, verbose=False, conf=0.25)
    
    detectado = False
    for r in results:
        for box in r.boxes:
            # Calibração: Classe 19 (Cow) e Confiança > 30%
            if int(box.cls) == 19 and float(box.conf) > 0.30: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calibração: Margem de 10% para melhor enquadramento do peso
                w, h = x2 - x1, y2 - y1
                pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                
                img_pil = img_pil.crop((
                    max(0, x1 - pad_w), 
                    max(0, y1 - pad_h), 
                    min(img_pil.width, x2 + pad_w), 
                    min(img_pil.height, y2 + pad_h)
                ))
                detectado = True
                break
                
    # Filtro Morfológico CLAHE para destacar contornos
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    res = clahe.apply(gray)
    final = cv2.cvtColor(res, cv2.GRAY2RGB)
    return final, detectado

# --- INTERFACE PRINCIPAL ---
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide", page_icon="🐂")
init_db()
yolo, ia_peso = load_models()

st.title("🐂 Rayvora Vision Pro: Inteligência de Campo")

# Diagnóstico Rápido no Sidebar
with st.sidebar:
    st.header("📊 Status do Sistema")
    if ia_peso: st.success("IA de Peso: ONLINE")
    else: st.error("IA de Peso: OFFLINE")
    
    conn = sqlite3.connect(str(DB_PATH))
    total = pd.read_sql("SELECT COUNT(*) as n FROM pesagens", conn).iloc[0]['n']
    conn.close()
    st.metric("Registros no Banco", f"{total} animais")

tabs = st.tabs(["🚀 Nova Pesagem", "📈 Histórico e GMD", "📦 Data Hub (Retreino)"])

# ABA 1: PESAGEM
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        brinco = st.text_input("ID do Animal:", "BOI_")
        up = st.file_uploader("Upload da foto (Vista Superior/Traseira)", type=['jpg', 'jpeg', 'png'])
        
    if up:
        img_raw = Image.open(up).convert('RGB')
        col1.image(img_raw, caption="Original", width=400) # Ajustado conforme log
        
        if st.button("🚀 Iniciar Análise Visual"):
            with st.spinner("IA processando morfologia..."):
                img_tratada, achou = pipeline_visao(img_raw, yolo)
                
                if not achou:
                    st.error("❌ O sistema não detectou um bovino com clareza. Tente outro ângulo.")
                else:
                    col2.image(img_tratada, caption="Segmentação Inteligente", width=400)
                    
                    if ia_peso:
                        # Preparação para o modelo H5 (128x128)
                        input_ia = cv2.resize(img_tratada, (128, 128)) / 255.0
                        peso = float(ia_peso.predict(np.expand_dims(input_ia, axis=0), verbose=False)[0][0])
                        
                        # Salva foto para retreino futuro
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_arquivo = f"{brinco}_{ts}.jpg"
                        img_raw.save(IMG_SAVE_PATH / nome_arquivo)
                        
                        # Salva no Banco de Dados
                        conn = sqlite3.connect(str(DB_PATH))
                        conn.execute("INSERT INTO pesagens (brinco_id, data, peso, foto_nome) VALUES (?,?,?,?)",
                                     (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso, nome_arquivo))
                        conn.commit()
                        conn.close()
                        
                        st.balloons()
                        st.success(f"✅ Pesagem Concluída: {peso:.2f} kg")

# ABA 2: ANALYTICS
with tabs[1]:
    st.subheader("Curva de Crescimento Diário")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM pesagens", conn)
    conn.close()
    
    if not df.empty:
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        selecionado = st.selectbox("Escolha o Brinco:", df['brinco_id'].unique())
        df_plot = df[df['brinco_id'] == selecionado].sort_values('data')
        
        fig = px.line(df_plot, x='data', y='peso', markers=True, title=f"Evolução de Peso - {selecionado}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum dado registrado até o momento.")

# ABA 3: DATA HUB
with tabs[2]:
    st.subheader("📦 Dataset para Retreinamento")
    st.write("Abaixo estão as últimas capturas validadas pela IA.")
    
    conn = sqlite3.connect(str(DB_PATH))
    regs = pd.read_sql("SELECT * FROM pesagens ORDER BY id DESC LIMIT 12", conn)
    conn.close()
    
    if not regs.empty:
        for i in range(0, len(regs), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(regs):
                    row = regs.iloc[i + j]
                    caminho_img = IMG_SAVE_PATH / str(row['foto_nome'])
                    if caminho_img.exists():
                        cols[j].image(str(caminho_img), caption=f"{row['brinco_id']}\n{row['peso']:.1f} kg")
    else:
        st.warning("A galeria de imagens será preenchida conforme você realizar as pesagens.")