import streamlit as st
import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from Pathlib import Path

# --- CONFIGURAÇÕES ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_SAVE_PATH = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

if not IMG_SAVE_PATH.exists():
    os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# --- MODELOS ---
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    try:
        ia_peso = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except:
        ia_peso = None
    return yolo, ia_peso

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  brinco_id TEXT, data TEXT, peso REAL, foto_nome TEXT)''')
    conn.commit()
    conn.close()

# --- INTERFACE ---
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide")
init_db()
yolo, ia_peso = load_models()

st.title("🐂 Rayvora Vision Pro: Refinamento de Biomassa")

abas = st.tabs(["🚀 Nova Pesagem", "📊 Histórico e GMD", "📦 Data Hub (Retreino)"])

with abas[0]:
    c1, c2 = st.columns(2)
    with c1:
        brinco = st.text_input("ID do Brinco:", "BOI_")
        up = st.file_uploader("Foto do Animal (Vista Traseira)", type=['jpg', 'jpeg', 'png'])
        
    if up:
        img_raw = Image.open(up).convert('RGB')
        c1.image(img_raw, caption="Original", use_container_width=True)
        
        if st.button("🚀 Validar e Calcular"):
            with st.spinner("Analisando..."):
                # 1. Filtro YOLO Rigoroso
                res = yolo(np.array(img_raw), verbose=False)
                detectado = False
                for r in res:
                    for box in r.boxes:
                        conf = float(box.conf)
                        if int(box.cls) == 19 and conf > 0.60: # 60% de confiança
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            img_raw = img_raw.crop((x1, y1, x2, y2))
                            detectado = True
                            break
                
                if not detectado:
                    st.error("❌ Erro: Nenhum bovino detectado com clareza. Certifique-se de que o animal está visível.")
                else:
                    # 2. Processamento CLAHE
                    img_np = np.array(img_raw)
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    img_viz = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)
                    c2.image(img_viz, caption="Recorte Inteligente (IA Vision)", use_container_width=True)
                    
                    if ia_peso:
                        # 3. Predição
                        prep = cv2.resize(img_viz, (128, 128)) / 255.0
                        peso = float(ia_peso.predict(np.expand_dims(prep, axis=0), verbose=False)[0][0])
                        
                        # 4. Salvar com Foto para Retreino
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_f = f"{brinco}_{ts}.jpg"
                        img_raw.save(IMG_SAVE_PATH / nome_f)
                        
                        conn = sqlite3.connect(str(DB_PATH))
                        conn.execute("INSERT INTO pesagens (brinco_id, data, peso, foto_nome) VALUES (?,?,?,?)",
                                     (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso, nome_f))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"✅ Peso Estimado: {peso:.2f} kg")

with abas[1]:
    st.subheader("Gráficos de Desempenho")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM pesagens", conn)
    conn.close()
    if not df.empty:
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        sel = st.selectbox("Selecione o Animal:", df['brinco_id'].unique())
        df_filt = df[df['brinco_id'] == sel].sort_values('data')
        st.plotly_chart(px.line(df_filt, x='data', y='peso', markers=True), use_container_width=True)
    else:
        st.info("Aguardando primeiras pesagens.")

with abas[2]:
    st.subheader("📦 Galeria de Refinamento")
    st.markdown("Use estas imagens para validar as predições e retreinar o modelo no futuro.")
    
    conn = sqlite3.connect(str(DB_PATH))
    registros = pd.read_sql("SELECT * FROM pesagens ORDER BY id DESC LIMIT 10", conn)
    conn.close()
    
    if not registros.empty:
        cols = st.columns(4)
        for i, row in registros.iterrows():
            idx = i % 4
            caminho_foto = IMG_SAVE_PATH / row['foto_nome']
            if caminho_foto.exists():
                cols[idx].image(str(caminho_foto), caption=f"{row['brinco_id']} - {row['peso']:.1f}kg")
    else:
        st.warning("Nenhuma imagem salva ainda.")