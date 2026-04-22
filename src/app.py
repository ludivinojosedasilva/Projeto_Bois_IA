import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import os

# --- CONFIGURAÇÕES DE CAMINHO ---
# No Streamlit Cloud, os caminhos são relativos à raiz do repositório
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# O banco de dados e as fotos ficam na estrutura do projeto
DB_PATH = os.path.join(BASE_DIR, 'monitoramento_bois.db')
IMG_SAVE_PATH = os.path.join(BASE_DIR, 'fotos_pesagens')
# Caminho para o modelo .h5 dentro da pasta 'models' (um nível acima de 'src')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_peso_bois.h5')

# Garantir que a pasta de fotos existe no servidor
if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)

def init_db():
    """Cria o banco de dados e a tabela se não existirem"""
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
    """Carrega o modelo Keras de forma otimizada para o Streamlit"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mae')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo do modelo: {e}")
        return None

def verificar_se_e_boi(img_input, modelo):
    """
    Filtro de consistência: Se o peso estimado for absurdo (fora de 50-1500kg),
    o sistema alerta que a imagem pode não ser de um bovino válido.
    """
    pred = modelo.predict(img_input)
    peso = float(pred[0][0])
    # Critério técnico para gado de corte adulto
    if peso < 50 or peso > 1500:
        return False, peso
    return True, peso

# --- INTERFACE DO USUÁRIO (UI) ---
st.set_page_config(page_title="Projeto Integrador - UFSC", layout="wide", page_icon="🐂")
init_db()

st.title("🐂 Monitoramento de Peso Bovino via IA")
st.markdown("Solução de Visão Computacional desenvolvida para o **Projeto Integrador - UFSC Araranguá**")

menu = ["Nova Pesagem", "Histórico e Auditoria"]
escolha = st.sidebar.selectbox("Navegação", menu)

if escolha == "Nova Pesagem":
    st.header("⚖️ Realizar Nova Estimativa")
    
    col_input, col_view = st.columns([1, 1])
    
    with col_input:
        brinco = st.text_input("Identificação do Animal (Brinco):", "BOI_")
        foto = st.file_uploader("Selecione a foto (Vista Traseira/Back View)", type=['jpg', 'jpeg', 'png'])
    
    if foto is not None:
        img_original = Image.open(foto).convert('RGB')
        with col_view:
            st.image(img_original, caption="Imagem carregada para análise", width=400)
        
        if st.button("🚀 Calcular Peso"):
            with st.spinner('Aguarde, a IA está processando os dados biométricos...'):
                model = load_model_ia()
                
                if model:
                    # Pré-processamento conforme o treinamento (128x128)
                    img_arr = np.array(img_original)
                    img_res = cv2.resize(img_arr, (128, 128)) / 255.0
                    img_input = np.expand_dims(img_res, axis=0)
                    
                    # Predição e Validação pelo Filtro
                    valido, peso_final = verificar_se_e_boi(img_input, model)
                    
                    if not valido:
                        st.error(f"❌ Imagem Rejeitada: O peso estimado de {peso_final:.2f}kg é inconsistente para um bovino.")
                    else:
                        # 1. Salvar Foto para Auditoria futura
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_arq_foto = f"{brinco}_{timestamp}.jpg"
                        caminho_final_foto = os.path.join(IMG_SAVE_PATH, nome_arq_foto)
                        img_original.save(caminho_final_foto)
                        
                        # 2. Registrar no Banco de Dados SQLite
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        c.execute("INSERT INTO pesagens (brinco_id, data, peso_estimado, caminho_foto) VALUES (?, ?, ?, ?)", 
                                  (brinco, agora, peso_final, nome_arq_foto))
                        conn.commit()
                        conn.close()
                        
                        st.success("✅ Pesagem registrada com sucesso!")
                        st.metric(label="Peso Estimado", value=f"{peso_final:.2f} kg")

elif escolha == "Histórico e Auditoria":
    st.header("📈 Histórico de Registros")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT id, brinco_id, data, peso_estimado, caminho_foto FROM pesagens ORDER BY id DESC", conn)
        conn.close()
        
        if not df.empty:
            # Exibição da tabela principal
            st.dataframe(df[['id', 'brinco_id', 'data', 'peso_estimado']], use_container_width=True)
            
            st.divider()
            st.subheader("🔍 Auditoria Visual e Download")
            
            id_audit = st.selectbox("Escolha o ID da pesagem para ver a evidência:", df['id'])
            
            # Recuperar dados da imagem selecionada
            linha = df[df['id'] == id_audit]
            nome_da_foto = linha['caminho_foto'].values[0]
            caminho_foto_audit = os.path.join(IMG_SAVE_PATH, nome_da_foto)
            
            if os.path.exists(caminho_foto_audit):
                st.image(Image.open(caminho_foto_audit), caption=f"Evidência da Pesagem #{id_audit}", width=500)
                
                # Botão de download da imagem
                with open(caminho_foto_audit, "rb") as img_file:
                    st.download_button(
                        label="📥 Baixar Foto de Auditoria",
                        data=img_file,
                        file_name=nome_da_foto,
                        mime="image/jpeg"
                    )
            else:
                st.warning("O arquivo de imagem foi removido ou não existe no servidor.")
            
            # --- SEÇÃO DE EXPORTAÇÃO DO BANCO DE DADOS ---
            st.divider()
            st.subheader("📦 Exportar Dados do Sistema")
            if os.path.exists(DB_PATH):
                with open(DB_PATH, "rb") as f:
                    st.download_button(
                        label="📥 Baixar Banco de Dados (.db)",
                        data=f,
                        file_name="monitoramento_bois.db",
                        mime="application/x-sqlite3"
                    )
        else:
            st.info("Nenhuma pesagem foi encontrada no banco de dados.")
    else:
        st.error("Erro: Banco de dados não inicializado.")