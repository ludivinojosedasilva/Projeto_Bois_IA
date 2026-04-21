import streamlit as st
import os
import sqlite3
import pandas as pd
import numpy as np
import cv2
import plotly.express as px
import tensorflow as tf
from PIL import Image
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURAÇÕES DE AMBIENTE E CAMINHOS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "monitoramento_bois.db"
IMG_DIR = BASE_DIR / "src" / "fotos_pesagens"
MODEL_PATH = BASE_DIR / "models" / "modelo_peso_bois.h5"

if not IMG_DIR.exists():
    os.makedirs(IMG_DIR, exist_ok=True)

# --- SISTEMA DE BANCO DE DADOS ---
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # Tabela principal
    cursor.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      brinco_id TEXT, data TEXT, peso REAL, foto_nome TEXT)''')
    
    # Migração automática para garantir que a coluna de fotos exista
    cursor.execute("PRAGMA table_info(pesagens)")
    colunas = [col[1] for col in cursor.fetchall()]
    if 'foto_nome' not in colunas:
        cursor.execute("ALTER TABLE pesagens ADD COLUMN foto_nome TEXT DEFAULT 'sem_foto.jpg'")
    
    conn.commit()
    conn.close()

# --- CARREGAMENTO DE INTELIGÊNCIA ARTIFICIAL ---
@st.cache_resource
def load_resources():
    yolo = YOLO('yolov8n.pt')
    try:
        weight_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as e:
        st.error(f"Erro ao carregar modelo .h5: {e}")
        weight_model = None
    return yolo, weight_model

# --- PIPELINE DE VISÃO COMPUTACIONAL ---
def processar_biometria(img_pil, _yolo):
    img_np = np.array(img_pil)
    results = _yolo(img_np, verbose=False, conf=0.3)
    
    detectado = False
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 19: # Classe 'cow' no COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Recorte com margem de 10% para escala
                w, h = x2 - x1, y2 - y1
                pw, ph = int(w * 0.1), int(h * 0.1)
                img_pil = img_pil.crop((
                    max(0, x1 - pw), max(0, y1 - ph), 
                    min(img_pil.width, x2 + pw), min(img_pil.height, y2 + ph)
                ))
                detectado = True
                break
                
    # Filtro Morfológico CLAHE
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    res = clahe.apply(gray)
    final = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB) # Correção COLOR_
    return final, detectado

# --- INTERFACE DO USUÁRIO (STREAMLIT) ---
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide", page_icon="🐂")
init_db()
yolo_net, weight_net = load_resources()

st.title("🐂 Rayvora Vision: Gestão Inteligente de Rebanho")

tabs = st.tabs(["🚀 Pesagem Direta", "📊 Dashboard", "📦 Data Hub"])

# ABA 1: OPERAÇÃO
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Entrada de Dados")
        id_animal = st.text_input("Identificação do Brinco:", "BOI_")
        arquivo = st.file_uploader("Capturar ou Carregar Foto", type=['jpg', 'jpeg', 'png'])
        
    if arquivo:
        imagem_original = Image.open(arquivo).convert('RGB')
        c1.image(imagem_original, caption="Imagem Original", use_container_width=True)
        
        if st.button("🚀 Executar Pesagem por IA"):
            with st.spinner("Analisando morfologia..."):
                img_final, achou = processar_biometria(imagem_original, yolo_net)
                
                if not achou:
                    st.warning("⚠️ Bovino não detectado com clareza. Tente outro ângulo.")
                else:
                    c2.subheader("Resultado da Análise")
                    c2.image(img_final, caption="Segmentação Biométrica", use_container_width=True)
                    
                    if weight_net:
                        # Inferência de Peso
                        input_tensor = cv2.resize(img_final, (128, 128)) / 255.0
                        input_tensor = np.expand_dims(input_tensor, axis=0)
                        predicao = weight_net.predict(input_tensor, verbose=False)[0][0]
                        
                        # Registro
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_foto = f"{id_animal}_{timestamp}.jpg"
                        imagem_original.save(IMG_DIR / nome_foto)
                        
                        conn = sqlite3.connect(str(DB_PATH))
                        conn.execute("INSERT INTO pesagens (brinco_id, data, peso, foto_nome) VALUES (?,?,?,?)",
                                     (id_animal, datetime.now().strftime("%d/%m/%Y %H:%M"), predicao, nome_foto))
                        conn.commit()
                        conn.close()
                        
                        c2.success(f"Peso Estimado: {predicao:.2f} kg")
                        st.balloons()

# ABA 2: ANALYTICS
with tabs[1]:
    st.subheader("Histórico de Crescimento")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM pesagens", conn)
    conn.close()
    
    if not df.empty:
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        animal_sel = st.selectbox("Selecione o Animal para Análise:", df['brinco_id'].unique())
        
        df_filt = df[df['brinco_id'] == animal_sel].sort_values('data')
        fig = px.line(df_filt, x='data', y='peso', markers=True, title=f"Evolução de Biomassa: {animal_sel}")
        st.plotly_chart(fig, use_container_width=True)
        st.table(df_filt[['data', 'peso']].tail(5))
    else:
        st.info("Aguardando os primeiros registros de campo.")

# ABA 3: DATA HUB (RETREINO)
with tabs[2]:
    st.subheader("📦 Repositório de Imagens para Refinamento")
    st.write("Estas imagens e pesos serão usados para calibrar a precisão do modelo Rayvora.")
    
    conn = sqlite3.connect(str(DB_PATH))
    registros = pd.read_sql("SELECT * FROM pesagens ORDER BY id DESC LIMIT 12", conn)
    conn.close()
    
    if not registros.empty:
        for i in range(0, len(registros), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(registros):
                    row = registros.iloc[i + j]
                    path_img = IMG_DIR / str(row['foto_nome'])
                    if path_img.exists():
                        cols[j].image(str(path_img), caption=f"{row['brinco_id']}\n{row['peso']:.1f} kg")
    else:
        st.warning("Galeria vazia.")