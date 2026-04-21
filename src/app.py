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

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'monitoramento_bois.db')
IMG_SAVE_PATH = os.path.join(BASE_DIR, 'fotos_pesagens')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_peso_bois.h5')

# Garantir existência de pastas
if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pesagens 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  brinco_id TEXT, 
                  data TEXT, 
                  peso_estimado REAL,
                  caminho_foto TEXT)''')
    conn.commit()
    conn.close()

@st.cache_resource
def load_model_ia():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mae')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def calcular_gmd(brinco_id, peso_atual):
    """Calcula o Ganho Médio Diário comparando com a última pesagem no banco"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT data, peso_estimado FROM pesagens WHERE brinco_id = ? ORDER BY id DESC LIMIT 1"
    df_ant = pd.read_sql_query(query, conn, params=(brinco_id,))
    conn.close()

    if not df_ant.empty:
        peso_ant = df_ant['peso_estimado'].values[0]
        data_ant = datetime.strptime(df_ant['data'].values[0], "%d/%m/%Y %H:%M:%S")
        hoje = datetime.now()
        
        dias = (hoje - data_ant).days
        if dias <= 0: dias = 1 # Evita divisão por zero
        
        gmd = (peso_atual - peso_ant) / dias
        return gmd, dias, peso_ant
    return None, None, None

def verificar_se_e_boi(img_input, modelo):
    pred = modelo.predict(img_input)
    peso = float(pred[0][0])
    # Filtro comercial: Recusa se o peso for impossível para a raça/idade
    if peso < 50 or peso > 1500:
        return False, peso
    return True, peso

# --- INTERFACE ---
st.set_page_config(page_title="Rayvora Analytics - UFSC", layout="wide", page_icon="📈")
init_db()

st.title("🐂 Rayvora: Inteligência em Ganho de Peso")
st.markdown("Solução Profissional para Substituição de Balanças Físicas")

menu = ["Nova Pesagem", "Histórico e Analytics"]
escolha = st.sidebar.selectbox("Navegação", menu)

if escolha == "Nova Pesagem":
    st.header("⚖️ Estimativa de Biomassa e GMD")
    col1, col2 = st.columns(2)
    
    with col1:
        brinco = st.text_input("ID do Animal (Brinco):", "BOI_")
        foto = st.file_uploader("Capturar ou Carregar Foto (Back View)", type=['jpg', 'jpeg', 'png'])
    
    if foto is not None:
        img_original = Image.open(foto).convert('RGB')
        col2.image(img_original, caption="Preview da Captura", width=350)
        
        if st.button("🚀 Processar e Registrar"):
            with st.spinner('Analisando morfologia do animal...'):
                model = load_model_ia()
                if model:
                    # Pre-processamento
                    img_arr = np.array(img_original)
                    img_res = cv2.resize(img_arr, (128, 128)) / 255.0
                    img_input = np.expand_dims(img_res, axis=0)
                    
                    valido, peso_final = verificar_se_e_boi(img_input, model)
                    
                    if not valido:
                        st.error(f"❌ Imagem Inconsistente: Peso estimado ({peso_final:.2f}kg) fora dos padrões.")
                    else:
                        # Cálculo do GMD (Ganho Médio Diário)
                        gmd, intervalo, peso_ant = calcular_gmd(brinco, peso_final)
                        
                        # Salvar arquivos
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_foto = f"{brinco}_{timestamp}.jpg"
                        img_original.save(os.path.join(IMG_SAVE_PATH, nome_foto))
                        
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        c.execute("INSERT INTO pesagens (brinco_id, data, peso_estimado, caminho_foto) VALUES (?, ?, ?, ?)", 
                                  (brinco, agora, peso_final, nome_foto))
                        conn.commit()
                        conn.close()

                        # Dashboard de Resultado Imediato
                        st.success(f"Pesagem Registrada: {peso_final:.2f} kg")
                        if gmd is not None:
                            c1, c2 = st.columns(2)
                            c1.metric("GMD Atual", f"{gmd:.3f} kg/dia", delta=f"{peso_final - peso_ant:.2f} kg")
                            c2.info(f"Último manejo deste animal foi há {intervalo} dias.")

elif escolha == "Histórico e Analytics":
    st.header("📊 Painel de Desempenho do Rebanho")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM pesagens", conn)
        conn.close()

        if not df.empty:
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            
            # Filtro por animal
            lista_bois = df['brinco_id'].unique()
            boi_sel = st.selectbox("Selecione um animal para análise individual:", lista_bois)
            
            df_boi = df[df['brinco_id'] == boi_sel].sort_values('data')
            
            # --- GRÁFICO DE EVOLUÇÃO ---
            fig = px.line(df_boi, x='data', y='peso_estimado', 
                          title=f'Curva de Crescimento - Animal {boi_sel}',
                          markers=True, line_shape='spline', render_mode="svg")
            fig.update_layout(yaxis_title="Peso (kg)", xaxis_title="Data")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- KPIS ---
            k1, k2, k3 = st.columns(3)
            k1.metric("Peso Inicial", f"{df_boi['peso_estimado'].iloc[0]:.2f} kg")
            k2.metric("Peso Atual", f"{df_boi['peso_estimado'].iloc[-1]:.2f} kg")
            total_ganho = df_boi['peso_estimado'].iloc[-1] - df_boi['peso_estimado'].iloc[0]
            k3.metric("Ganho Total no Ciclo", f"{total_ganho:.2f} kg")

            st.divider()
            st.subheader("📋 Auditoria de Registros")
            st.dataframe(df_boi[['id', 'data', 'peso_estimado', 'caminho_foto']], use_container_width=True)
            
            # Download do Banco para relatório UFSC
            st.divider()
            with open(DB_PATH, "rb") as f:
                st.download_button("📥 Baixar Base de Dados (.db)", f, "monitoramento.db")
        else:
            st.info("Nenhum dado registrado ainda.")