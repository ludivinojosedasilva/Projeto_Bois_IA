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
from pathlib import Path

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_SAVE_PATH = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

# Garante que a pasta de fotos para o Data Hub exista
if not IMG_SAVE_PATH.exists():
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# --- BANCO DE DADOS COM MIGRAÇÃO AUTOMÁTICA ---
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # Cria tabela básica se não existir
    cursor.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      brinco_id TEXT, 
                      data TEXT, 
                      peso REAL)''')
    
    # Migração: Verifica se a coluna 'foto_nome' existe
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
            st.error(f"Erro ao carregar IA de Peso: {e}")
    return yolo, ia_peso

# --- PIPELINE DE VISÃO COM DEPURAÇÃO VISUAL ---
def pipeline_visao(img_pil, _yolo):
    img_np = np.array(img_pil)
    # Calibração de detecção inicial
    results = _yolo(img_np, verbose=False, conf=0.25)
    
    melhor_box = None
    maior_area = 0
    conf_detect = 0
    
    # Seleciona a maior detecção de "cow" (Classe 19) para evitar confusão com fundo
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > maior_area:
                    maior_area = area
                    melhor_box = (x1, y1, x2, y2)
                    conf_detect = float(box.conf)

    if melhor_box:
        x1, y1, x2, y2 = melhor_box
        
        # Feedback Visual: Desenha o retângulo verde no que a IA identificou
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(img_np, f"BOVINO {conf_detect:.1%}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Recorte com margem de segurança de 10%
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * 0.1), int(h * 0.1)
        
        img_recorte = img_pil.crop((
            max(0, x1 - pad_w), 
            max(0, y1 - pad_h), 
            min(img_pil.width, x2 + pad_w), 
            min(img_pil.height, y2 + pad_h)
        ))
        
        # Filtro Morfológico CLAHE
        rec_np = np.array(img_recorte)
        gray = cv2.cvtColor(rec_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        res_clahe = clahe.apply(gray)
        final_ia = cv2.cvtColor(res_clahe, cv2.COLOR_GRAY2RGB)
        
        return img_np, final_ia, True, conf_detect
    
    return img_np, None, False, 0

# --- INTERFACE PRINCIPAL ---
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide", page_icon="🐂")
init_db()
yolo, ia_peso = load_models()

st.title("🐂 Rayvora Vision Pro: Gestão e Biometria")

# SIDEBAR DE CONTROLE E CALIBRAÇÃO
with st.sidebar:
    st.header("📊 Status e Calibração")
    if ia_peso: st.success("IA de Peso: ONLINE")
    else: st.error("IA de Peso: OFFLINE")
    
    st.divider()
    st.subheader("⚙️ Ajustes de Campo")
    st.info("Use estes controles para calibrar o erro sistemático do modelo.")
    
    bias_ajuste = st.slider("Ajuste de Viés (kg)", -100.0, 100.0, 0.0, 
                             help="Soma ou subtrai um valor fixo do peso final.")
    fator_escala = st.slider("Fator de Escala (%)", 80, 120, 100) / 100.0
    
    st.divider()
    conn = sqlite3.connect(str(DB_PATH))
    total = pd.read_sql("SELECT COUNT(*) as n FROM pesagens", conn).iloc[0]['n']
    conn.close()
    st.metric("Total de Registros", f"{total} bois")

tabs = st.tabs(["🚀 Nova Pesagem", "📈 Evolução Animal", "📦 Data Hub (Retreino)"])

# ABA 1: PESAGEM
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        brinco = st.text_input("Identificação (Brinco):", "BOI_")
        up = st.file_uploader("Foto do animal", type=['jpg', 'jpeg', 'png'])
        
    if up:
        img_raw = Image.open(up).convert('RGB')
        
        if st.button("🚀 Iniciar Análise Visual"):
            with st.spinner("IA processando biomassa..."):
                img_debug, img_ia, achou, conf = pipeline_visao(img_raw, yolo)
                
                # Exibe a detecção com o retângulo verde
                col1.image(img_debug, caption=f"Identificação YOLO ({conf:.1%})", width=450)
                
                if not achou:
                    st.error("❌ Nenhum bovino identificado com confiança suficiente.")
                else:
                    # Exibe o que foi enviado para a rede de peso
                    col2.image(img_ia, caption="Segmentação para Estimativa de Peso", width=450)
                    
                    if ia_peso:
                        # Inferência
                        input_ia = cv2.resize(img_ia, (128, 128)) / 255.0
                        peso_bruto = float(ia_peso.predict(np.expand_dims(input_ia, axis=0), verbose=False)[0][0])
                        
                        # Aplicação da Calibração
                        peso_final = (peso_bruto * fator_escala) + bias_ajuste
                        
                        # Salvamento da imagem e dados
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_f = f"{brinco}_{ts}.jpg"
                        img_raw.save(IMG_SAVE_PATH / nome_f)
                        
                        conn = sqlite3.connect(str(DB_PATH))
                        conn.execute("INSERT INTO pesagens (brinco_id, data, peso, foto_nome) VALUES (?,?,?,?)",
                                     (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso_final, nome_f))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"✅ Peso Estimado: {peso_final:.2f} kg")
                        st.caption(f"Peso Bruto IA: {peso_bruto:.1f}kg | Ajuste de Viés: {bias_ajuste}kg")
                        if conf < 0.5:
                            st.warning("⚠️ Atenção: Baixa confiança na detecção. O peso pode oscilar.")

# ABA 2: ANALYTICS
with tabs[1]:
    st.subheader("Histórico de Ganho de Peso")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM pesagens", conn)
    conn.close()
    
    if not df.empty:
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        sel = st.selectbox("Selecione o Animal:", df['brinco_id'].unique())
        df_filt = df[df['brinco_id'] == sel].sort_values('data')
        
        fig = px.line(df_filt, x='data', y='peso', markers=True, title=f"Curva de Peso - {sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum dado registrado.")

# ABA 3: DATA HUB
with tabs[2]:
    st.subheader("📦 Galeria de Coleta de Dados")
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