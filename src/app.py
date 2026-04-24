import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'monitoramento_bois.db')
IMG_SAVE_PATH = os.path.join(BASE_DIR, 'fotos_pesagens')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_peso_bois.h5')

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
            peso_real REAL,
            confianca REAL,
            erro REAL,
            caminho_foto TEXT
        )
    ''')

    c.execute("CREATE INDEX IF NOT EXISTS idx_brinco ON pesagens(brinco_id)")
    conn.commit()
    conn.close()

# ==============================
# MODEL
# ==============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mae')
    return model

# ==============================
# IMAGE PROCESSING (CLAHE)
# ==============================
def preprocess_image(img):
    img = np.array(img)
    img = cv2.resize(img, (128, 128))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return final / 255.0

# ==============================
# CONFIDENCE FIXED
# ==============================
def calculate_confidence(std, mean):
    if mean == 0:
        return 0.0

    coef_var = std / abs(mean)
    confidence = np.exp(-coef_var * 5) * 100
    return float(np.clip(confidence, 0, 100))

# ==============================
# MULTI-INFERENCE
# ==============================
def predict_with_confidence(model, img, n=10):
    preds = []

    for _ in range(n):
        noise = np.random.normal(0, 0.01, img.shape)
        img_noisy = np.clip(img + noise, 0, 1)

        inp = np.expand_dims(img_noisy, axis=0)
        pred = model.predict(inp, verbose=0)[0][0]
        preds.append(pred)

    mean = np.mean(preds)
    std = np.std(preds)

    confidence = calculate_confidence(std, mean)
    error = std * 2

    return float(mean), float(confidence), float(error)

# ==============================
# VALIDATION
# ==============================
def validar_peso(peso):
    return 50 <= peso <= 1500

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide")
init_db()

st.title("🐂 Rayvora Vision Pro")
st.markdown("Sistema Inteligente de Estimativa de Peso Bovino")

menu = ["Nova Pesagem", "Histórico", "Dashboard"]
escolha = st.sidebar.selectbox("Menu", menu)

modo_mobile = st.sidebar.checkbox("Modo Mobile")

# ==============================
# NOVA PESAGEM
# ==============================
if escolha == "Nova Pesagem":

    if modo_mobile:
        col1 = st.container()
        col2 = st.container()
    else:
        col1, col2 = st.columns(2)

    with col1:
        brinco = st.text_input("Brinco", "BOI_")
        peso_real_input = st.number_input("Peso real (opcional)", min_value=0.0, step=1.0)
        foto = st.file_uploader("Imagem", type=["jpg","png","jpeg"])

    if foto:
        img = Image.open(foto).convert("RGB")

        with col2:
            st.image(img, use_container_width=True)

        if st.button("🚀 Calcular Peso (Alta Precisão)", use_container_width=True):

            with st.spinner("Processando..."):

                model = load_model()
                processed = preprocess_image(img)

                peso, conf, erro = predict_with_confidence(model, processed)

                if not validar_peso(peso):
                    st.error("Imagem inválida")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_img = f"{brinco}_{timestamp}.jpg"
                    path_img = os.path.join(IMG_SAVE_PATH, nome_img)
                    img.save(path_img)

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()

                    data = datetime.now().strftime("%d/%m/%Y %H:%M")

                    c.execute("""
                        INSERT INTO pesagens 
                        (brinco_id, data, peso_estimado, peso_real, confianca, erro, caminho_foto)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        brinco,
                        data,
                        peso,
                        peso_real_input if peso_real_input > 0 else None,
                        conf,
                        erro,
                        nome_img
                    ))

                    conn.commit()
                    conn.close()

                    st.success("Pesagem registrada!")

                    colA, colB, colC = st.columns(3)
                    colA.metric("Peso", f"{peso:.2f} kg")
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
        st.dataframe(df, use_container_width=True)

        selected_id = st.selectbox("Selecionar ID", df['id'])
        row = df[df['id'] == selected_id].iloc[0]

        img_path = os.path.join(IMG_SAVE_PATH, row['caminho_foto'])

        if os.path.exists(img_path):
            st.image(img_path, width=500)

        st.write(f"Peso: {row['peso_estimado']:.2f} kg")
        st.write(f"Confiança: {row['confianca']:.2f}%")
        st.write(f"Erro: ±{row['erro']:.2f} kg")

# ==============================
# DASHBOARD
# ==============================
elif escolha == "Dashboard":

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pesagens", conn)
    conn.close()

    df_valid = df.dropna(subset=['peso_real'])

    if not df_valid.empty:

        mae = np.mean(np.abs(df_valid['peso_real'] - df_valid['peso_estimado']))
        rmse = np.sqrt(np.mean((df_valid['peso_real'] - df_valid['peso_estimado'])**2))

        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mae:.2f} kg")
        col2.metric("RMSE", f"{rmse:.2f} kg")

        st.subheader("IA vs Peso Real")

        fig, ax = plt.subplots()

        ax.scatter(df_valid['peso_real'], df_valid['peso_estimado'])

        min_val = df_valid['peso_real'].min()
        max_val = df_valid['peso_real'].max()

        ax.plot([min_val, max_val], [min_val, max_val])

        ax.set_xlabel("Peso Real")
        ax.set_ylabel("Peso Estimado")

        st.pyplot(fig)

    else:
        st.info("Adicione pesos reais para gerar métricas.")