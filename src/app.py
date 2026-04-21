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

# --- CONFIGURAÇÕES ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_SAVE_PATH = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

if not IMG_SAVE_PATH.exists():
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# --- BANCO DE DADOS ---
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      brinco_id TEXT, data TEXT, peso REAL, foto_nome TEXT)''')
    conn.commit()
    conn.close()

# --- MODELOS ---
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    ia_peso = None
    if MODEL_PATH.exists():
        try:
            ia_peso = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        except Exception as e:
            st.error(f"Erro no modelo H5: {e}")
    return yolo, ia_peso

# --- PIPELINE DE VISÃO COM FEEDBACK VISUAL ---
def pipeline_visao(img_pil, _yolo):
    img_np = np.array(img_pil)
    results = _yolo(img_np, verbose=False, conf=0.25)
    
    melhor_box = None
    maior_area = 0
    conf_detect = 0
    
    # Busca pela maior detecção de "cow" (Classe 19)
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
        
        # Desenha a detecção para feedback do usuário
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(img_np, f"BOI {conf_detect:.1%}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Recorte com margem de 10%
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * 0.1), int(h * 0.1)
        
        img_recorte = img_pil.crop((
            max(0, x1 - pad_w), 
            max(0, y1 - pad_h), 
            min(img_pil.width, x2 + pad_w), 
            min(img_pil.height, y2 + pad_h)
        ))
        
        # Processamento CLAHE no recorte
        rec_np = np.array(img_recorte)
        gray = cv2.cvtColor(rec_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        res_clahe = clahe.apply(gray)
        final_ia = cv2.cvtColor(res_clahe, cv2.COLOR_GRAY2RGB)
        
        return img_np, final_ia, True, conf_detect
    
    return img_np, None, False, 0

# --- INTERFACE ---
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide", page_icon="🐂")
init_db()
yolo, ia_peso = load_models()

st.title("🐂 Rayvora Vision Pro: Diagnóstico de Pesagem")

tabs = st.tabs(["🚀 Nova Pesagem", "📈 Histórico", "📦 Data Hub"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        brinco = st.text_input("ID do Animal:", "BOI_")
        up = st.file_uploader("Upload da foto", type=['jpg', 'jpeg', 'png'])
        
    if up:
        img_raw = Image.open(up).convert('RGB')
        
        if st.button("🚀 Analisar e Pesar"):
            with st.spinner("IA processando..."):
                img_debug, img_ia, achou, conf = pipeline_visao(img_raw, yolo)
                
                c1.image(img_debug, caption=f"Detecção YOLO (Confiança: {conf:.1%})", width=450)
                
                if not achou:
                    st.error("❌ Animal não identificado. Verifique se o boi está visível e centralizado.")
                else:
                    c2.image(img_ia, caption="Recorte enviado para a rede neural de peso", width=450)
                    
                    if ia_peso:
                        prep = cv2.resize(img_ia, (128, 128)) / 255.0
                        peso = float(ia_peso.predict(np.expand_dims(prep, axis=0), verbose=False)[0][0])
                        
                        # Salvar
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_f = f"{brinco}_{ts}.jpg"
                        img_raw.save(IMG_SAVE_PATH / nome_f)
                        
                        conn = sqlite3.connect(str(DB_PATH))
                        conn.execute("INSERT INTO pesagens (brinco_id, data, peso, foto_nome) VALUES (?,?,?,?)",
                                     (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso, nome_f))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"✅ Peso Estimado: {peso:.2f} kg")
                        if conf < 0.5:
                            st.warning("⚠️ Atenção: Baixa confiança na detecção. O peso pode estar impreciso.")

# (Abas de Histórico e Data Hub seguem a mesma lógica anterior...)