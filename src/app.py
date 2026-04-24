import streamlit as st
import tensorflow as tf
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
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_v2.h5')  # pronto pro v2

os.makedirs(IMG_SAVE_PATH, exist_ok=True)

# ==============================
# DATABASE (COM MIGRAÇÃO SEGURA)
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

    c.execute("CREATE INDEX IF NOT EXISTS idx_brinco ON pesagens(brinco_id)")

    conn.commit()
    conn.close()

# ==============================
# MODEL (CACHE + ROBUSTEZ)
# ==============================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mae')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# ==============================
# IMAGE PROCESSING (PDI AVANÇADO)
# ==============================
def preprocess_image(img):
    img = np.array(img)
    img = cv2.resize(img, (128, 128))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # leve suavização (remove ruído)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img / 255.0

# ==============================
# CONFIDENCE MELHORADA
# ==============================
def calculate_confidence(std, mean):
    if mean == 0:
        return 0.0

    coef_var = std / abs(mean)
    confidence = np.exp(-coef_var * 5) * 100

    return float(np.clip(confidence, 0, 100))

# ==============================
# MULTI-INFERENCE (PRECISÃO MÁXIMA)
# ==============================
def predict_with_confidence(model, img, n=12):
    preds = []

    for _ in range(n):
        noise = np.random.normal(0, 0.008, img.shape)
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
# VALIDAÇÃO
# ==============================
def validar_peso(peso):
    return 50 <= peso <= 1500

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Rayvora Vision Pro", layout="wide")
init_db()

st.title("🐂 Rayvora Vision Pro")
st.markdown("Sistema Inteligente de Estimativa de Peso Bovino via IA")

menu = ["Nova Pesagem", "Histórico"]
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
        brinco = st.text_input("Brinco do animal", "BOI_")
        foto = st.file_uploader("Imagem", type=["jpg", "jpeg", "png"])

    if foto:
        img = Image.open(foto).convert("RGB")

        with col2:
            st.image(img, width='stretch')

        if st.button("🚀 Calcular Peso (Alta Precisão)", width='stretch'):

            with st.spinner("IA analisando características biométricas..."):

                model = load_model()

                if model is None:
                    st.stop()

                processed = preprocess_image(img)
                peso, conf, erro = predict_with_confidence(model, processed)

                if not validar_peso(peso):
                    st.error("Imagem inválida ou fora do padrão bovino")
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

                    st.success("Pesagem registrada com sucesso!")

                    colA, colB, colC = st.columns(3)
                    colA.metric("Peso", f"{peso:.2f} kg")
                    colB.metric("Confiança", f"{conf:.1f}%")
                    colC.metric("Margem de erro", f"±{erro:.2f} kg")

# ==============================
# HISTÓRICO
# ==============================
elif escolha == "Histórico":

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pesagens ORDER BY id DESC", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df, width='stretch')

        selected_id = st.selectbox("Selecionar ID", df['id'])
        row = df[df['id'] == selected_id].iloc[0]

        img_path = os.path.join(IMG_SAVE_PATH, row['caminho_foto'])

        if os.path.exists(img_path):
            st.image(img_path, width=500)

        st.write(f"Peso: {row['peso_estimado']:.2f} kg")
        st.write(f"Confiança: {row['confianca']:.2f}%")
        st.write(f"Erro: ±{row['erro']:.2f} kg")

    else:
        st.info("Nenhum registro encontrado.")