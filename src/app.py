import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import os
import plotly.express as px
from ultralytics import YOLO

# --- 1. CONFIGURAÇÕES DE AMBIENTE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'monitoramento_bois.db')
IMG_SAVE_PATH = os.path.join(BASE_DIR, 'fotos_pesagens')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_peso_bois.h5')

# Garante que a infraestrutura de pastas existe
if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)

# --- 2. CARREGAMENTO INTELIGENTE DE MODELOS (CACHE) ---

@st.cache_resource
def load_yolo_model():
    """Carrega o YOLOv8n (Nano) - Otimizado para execução em CPU/Cloud"""
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_weight_model():
    """Carrega o modelo Keras treinado para estimativa de biomassa"""
    try:
        # Carregamos sem compilar para evitar erros de versão de otimizadores
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mae')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o cérebro da IA (H5): {e}")
        return None

# --- 3. PIPELINE DE VISÃO COMPUTACIONAL RAYVORA ---

def pipeline_yolo_crop(img_pil, _model):
    """Detecta a presença do animal e isola o objeto de interesse"""
    img_np = np.array(img_pil)
    results = _model(img_np, verbose=False)
    
    for r in results:
        for box in r.boxes:
            # Classe 19 (vaca) no dataset COCO
            if int(box.cls) == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Margem de segurança de 5% para não cortar partes do corpo
                w, h = x2 - x1, y2 - y1
                x1 = max(0, x1 - int(w*0.05))
                y1 = max(0, y1 - int(h*0.05))
                x2 = min(img_np.shape[1], x2 + int(w*0.05))
                y2 = min(img_np.shape[0], y2 + int(h*0.05))
                
                return img_pil.crop((x1, y1, x2, y2)), True
    return img_pil, False

def pipeline_clahe_visual(img_pil):
    """Aplica equalização de histograma para neutralizar luz e sombra"""
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # CLAHE para realçar texturas musculares
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_equalizada = clahe.apply(gray)
    
    return cv2.cvtColor(img_equalizada, cv2.COLOR_GRAY2RGB)

# --- 4. LÓGICA DE DADOS E GMD ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  brinco_id TEXT, data TEXT, peso_estimado REAL, caminho_foto TEXT)''')
    conn.commit()
    conn.close()

def processar_gmd(brinco_id, peso_novo):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT data, peso_estimado FROM pesagens WHERE brinco_id = ? ORDER BY id DESC LIMIT 1"
    df_old = pd.read_sql_query(query, conn, params=(brinco_id,))
    conn.close()

    if not df_old.empty:
        peso_old = df_old['peso_estimado'].values[0]
        data_old = datetime.strptime(df_old['data'].values[0], "%d/%m/%Y %H:%M:%S")
        dias = (datetime.now() - data_old).days
        dias = dias if dias > 0 else 1
        return (peso_novo - peso_old) / dias, dias, peso_old
    return None, None, None

# --- 5. INTERFACE DO USUÁRIO (STREAMLIT) ---

st.set_page_config(page_title="Rayvora Pro - UFSC", layout="wide", page_icon="🐂")
init_db()

st.title("🐂 Rayvora Pro: Pesagem Digital Inteligente")
st.markdown("Monitoramento Bovino via YOLOv8 e Visão Computacional")

menu = ["🚀 Nova Pesagem", "📈 Dashboard & Auditoria"]
escolha = st.sidebar.selectbox("Módulos", menu)

if escolha == "🚀 Nova Pesagem":
    st.header("⚖️ Sistema de Captura e Estimativa")
    
    c_config, c_orig, c_crop, c_ia = st.columns([1, 1, 1, 1])
    
    with c_config:
        brinco = st.text_input("ID do Animal:", placeholder="Ex: BOI_55")
        foto = st.file_uploader("Upload da Imagem de Campo", type=['jpg', 'png', 'jpeg'])
    
    if foto:
        img_raw = Image.open(foto).convert('RGB')
        c_orig.image(img_raw, caption="1. Original", use_container_width=True)
        
        if st.button("Executar Pipeline Rayvora"):
            with st.spinner('Processando...'):
                # FASE A: Deteção e Recorte
                yolo = load_yolo_model()
                img_crop, detectou = pipeline_yolo_crop(img_raw, yolo)
                c_crop.image(img_crop, caption="2. Recorte YOLO" if detectou else "Não Detectado", use_container_width=True)
                
                # FASE B: Tratamento Visual CLAHE
                img_viz = pipeline_clahe_visual(img_crop)
                c_ia.image(img_viz, caption="3. Visão da IA (Contraste)", use_container_width=True)
                
                # FASE C: Inferência de Peso
                ia_peso = load_weight_model()
                img_input = cv2.resize(img_viz, (128, 128)) / 255.0
                pred = ia_peso.predict(np.expand_dims(img_input, axis=0), verbose=False)
                peso_final = float(pred[0][0])
                
                # Validação de Faixa Comercial
                if 40 < peso_final < 1800:
                    gmd, tempo, p_ant = processar_gmd(brinco, peso_final)
                    
                    # Salvar arquivos e Banco
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_f = f"{brinco}_{ts}.jpg"
                    img_raw.save(os.path.join(IMG_SAVE_PATH, nome_f))
                    
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("INSERT INTO pesagens (brinco_id, data, peso_estimado, caminho_foto) VALUES (?,?,?,?)",
                                 (brinco, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), peso_final, nome_f))
                    conn.commit()
                    conn.close()
                    
                    st.success(f"✅ Pesagem registrada: {peso_final:.2f} kg")
                    if gmd is not None:
                        m1, m2 = st.columns(2)
                        m1.metric("GMD Atual", f"{gmd:.3f} kg/dia", delta=f"{peso_final - p_ant:.2f} kg")
                        m2.info(f"Última pesagem há {tempo} dias.")
                    st.balloons()
                else:
                    st.error("❌ Erro: Peso estimado incompatível com perfil bovino.")

elif escolha == "📈 Dashboard & Auditoria":
    st.header("📊 Painel Analítico do Rebanho")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM pesagens", conn)
        conn.close()

        if not df.empty:
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            boi_sel = st.selectbox("Selecione o Animal:", df['brinco_id'].unique())
            df_boi = df[df['brinco_id'] == boi_sel].sort_values('data')
            
            # Gráfico Plotly
            fig = px.line(df_boi, x='data', y='peso_estimado', title=f"Histórico: {boi_sel}", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            # Botões de Exportação
            c1, c2 = st.columns(2)
            with open(DB_PATH, "rb") as f:
                c1.download_button("📥 Baixar Base de Dados (.db)", f, "rayvora_data.db")
            
            csv = df_boi.to_csv(index=False).encode('utf-8')
            c2.download_button("📥 Baixar CSV do Animal", csv, f"hist_{boi_sel}.csv", "text/csv")
        else:
            st.info("Nenhum dado registrado.")