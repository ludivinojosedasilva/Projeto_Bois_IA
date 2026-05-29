import streamlit as st
import ai_edge_litert.interpreter as litert
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import os

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'monitoramento_bois.db')
IMG_SAVE_PATH = os.path.join(BASE_DIR, 'fotos_pesagens')
# Atualizado para buscar o arquivo .tflite do Edge Impulse
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'model.tflite')

os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# ==============================
# DATABASE
# ==============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS pesagens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brinco_id TEXT,
            data TEXT,
            peso_estimado REAL,
            caminho_foto TEXT
        )
    ''')

    def add_column_if_not_exists(name, dtype):
        c.execute("PRAGMA table_info(pesagens)")
        columns = [col[1] for col in c.fetchall()]
        if name not in columns:
            c.execute(f"ALTER TABLE pesagens ADD COLUMN {name} {dtype}")

    add_column_if_not_exists("confianca", "REAL")
    add_column_if_not_exists("erro", "REAL")

    conn.commit()
    conn.close()

# ==============================
# MODEL (MIGRAÇÃO PARA LITER_RT / TFLITE)
# ==============================
@st.cache_resource
def load_model():
    try:
        # Inicializa o interpreter portátil do LiteRT
        interpreter = litert.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Erro crítico ao carregar o modelo em '{MODEL_PATH}': {e}")
        st.info("Verifique se o arquivo 'model.tflite' está na pasta correspondente.")
        return None

# ==============================
# IMAGE PROCESSING
# ==============================
def preprocess_image(img):
    img = np.array(img)
    # Redimensionamento padrão exigido pela CNN do Edge Impulse
    img = cv2.resize(img, (128, 128))

    # Preservação do espaço de cores e tratamento CLAHE original do seu projeto
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Normalização (1./255) alinhada com as features do Edge Impulse
    return img.astype(np.float32) / 255.0

# ==============================
# CONFIDENCE
# ==============================
def calculate_confidence(std, mean):
    if mean == 0:
        return 0.0

    coef_var = std / abs(mean)
    confidence = np.exp(-coef_var * 5) * 100

    return float(np.clip(confidence, 0, 100))

# ==============================
# MULTI-INFERENCE (LITER_RT)
# ==============================
def predict_with_confidence(interpreter, img, n=12):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    preds = []

    for _ in range(n):
        # Injeção de ruído estocástico gaussiano para simular incerteza
        noise = np.random.normal(0, 0.008, img.shape)
        img_noisy = np.clip(img + noise, 0, 1).astype(np.float32)

        # Configura o tensor de entrada adicionando o lote (Batch dimension)
        inp = np.expand_dims(img_noisy, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        
        # Coleta o peso predito pela camada de regressão linear
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        preds.append(pred)

    mean = np.mean(preds)
    std = np.std(preds)

    confidence = calculate_confidence(std, mean)
    error = std * 2

    return float(mean), float(confidence), float(error)

# ==============================
# FORMATADOR PROFISSIONAL
# ==============================
def formatar_peso(peso):
    return f"{peso:,.2f} kg".replace(",", ".")

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Rayvora Vision Pro v2", layout="wide")
init_db()

st.title("🐂 Rayvora Vision Pro - v2.0.0")
st.markdown("Sistema Inteligente de Estimativa de Peso Bovino via IA (Edge Impulse Engine)")

menu = ["Nova Pesagem", "Histórico"]
escolha = st.sidebar.selectbox("Menu", menu)

# ==============================
# NOVA PESAGEM
# ==============================
if escolha == "Nova Pesagem":

    col1, col2 = st.columns(2)

    with col1:
        brinco = st.text_input("Brinco do animal", "BOI_")
        foto = st.file_uploader("Imagem", type=["jpg", "jpeg", "png"])

    if foto:
        img = Image.open(foto).convert("RGB")

        with col2:
            st.image(img, width=400) # Ajustado largura fixa segura para layouts modernos do Streamlit

        if st.button("🚀 Calcular Peso"):

            with st.spinner("IA analisando imagem pelo motor LiteRT..."):

                interpreter = load_model()
                
                if interpreter is not None:
                    processed = preprocess_image(img)

                    peso, conf, erro = predict_with_confidence(interpreter, processed)

                    # salvar imagem localmente
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_img = f"{brinco}_{timestamp}.jpg"
                    path_img = os.path.join(IMG_SAVE_PATH, nome_img)
                    img.save(path_img)

                    # persistência no banco SQLite
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()

                    data = datetime.now().strftime("%d/%m/%Y %H:%M")

                    c.execute("""
                        INSERT INTO pesagens 
                        (brinco_id, data, peso_estimado, confianca, erro, caminho_foto)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        brinco,
                        data,
                        peso,
                        conf,
                        erro,
                        nome_img
                    ))

                    conn.commit()
                    conn.close()

                    st.success("Pesagem registrada com sucesso no sistema!")

                    colA, colB, colC = st.columns(3)
                    colA.metric("Peso", formatar_peso(peso))
                    colB.metric("Confiança", f"{conf:.1f}%")
                    colC.metric("Erro", f"±{erro:.2f} kg")

# ==============================
# HISTÓRICO
# ==============================
elif escolha == "Histórico":

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pesagens ORDER BY id DESC", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df, use_container_width=True) # Atualizado parâmetro obsoleto para o Streamlit estável
    else:
        st.info("Nenhum registro encontrado no histórico.")