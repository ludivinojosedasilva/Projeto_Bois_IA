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
# No Streamlit Cloud, os caminhos são relativos ao diretório raiz do repositório
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# O banco de dados e as fotos ficarão na pasta do projeto
DB_PATH = os.path.join(BASE_DIR, 'monitoramento_bois.db')
IMG_SAVE_PATH = os.path.join(BASE_DIR, 'fotos_pesagens')
# Caminho para o modelo dentro da pasta 'models'
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_peso_bois.h5')

# Garantir que a pasta de fotos existe
if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)

def init_db():
    """Inicializa o banco de dados SQLite"""
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
    """Carrega o modelo treinado de forma otimizada"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mae')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def verificar_se_e_boi(img_input, modelo):
    """
    Filtro de validação: Verifica se a predição faz sentido para um bovino.
    Serve como uma camada de segurança antes de registrar no banco.
    """
    pred = modelo.predict(img_input)
    peso = float(pred[0][0])
    # Critério: Peso deve estar entre 50kg e 1500kg para ser considerado válido
    if peso < 50 or peso > 1500:
        return False, peso
    return True, peso

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Projeto Integrador - UFSC", layout="wide")
init_db()

st.title("🐂 Monitoramento de Peso Bovino")
st.markdown("Protótipo de Pesagem Visual para a disciplina de Projeto Integrador")

menu = ["Nova Pesagem", "Histórico e Auditoria"]
escolha = st.sidebar.selectbox("Menu de Navegação", menu)

if escolha == "Nova Pesagem":
    st.header("⚖️ Realizar Nova Pesagem")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        brinco = st.text_input("Identificação do Animal (Brinco/ID):", "ID_")
        foto = st.file_uploader("Carregar foto da traseira (Back View)", type=['jpg', 'jpeg', 'png'])
    
    if foto is not None:
        img_original = Image.open(foto).convert('RGB')
        with col2:
            st.image(img_original, caption="Imagem carregada", width=400)
        
        if st.button("🚀 Calcular Peso"):
            with st.spinner('A IA está analisando a imagem...'):
                model = load_model_ia()
                if model:
                    # Pré-processamento (128x128 conforme o treinamento)
                    img_arr = np.array(img_original)
                    img_res = cv2.resize(img_arr, (128, 128)) / 255.0
                    img_input = np.expand_dims(img_res, axis=0)
                    
                    # Validação e Predição
                    valido, peso_final = verificar_se_e_boi(img_input, model)
                    
                    if not valido:
                        st.error(f"❌ Imagem Inválida: O sistema estimou {peso_final:.2f}kg, o que não condiz com um bovino adulto.")
                    else:
                        # 1. Salvar Foto Fisicamente para Auditoria
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_arquivo_foto = f"{brinco}_{timestamp}.jpg"
                        caminho_completo_foto = os.path.join(IMG_SAVE_PATH, nome_arquivo_foto)
                        img_original.save(caminho_completo_foto)
                        
                        # 2. Registrar no Banco de Dados
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        c.execute("INSERT INTO pesagens (brinco_id, data, peso_estimado, caminho_foto) VALUES (?, ?, ?, ?)", 
                                  (brinco, agora, peso_final, nome_arquivo_foto))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"✅ Pesagem Registrada com Sucesso!")
                        st.metric("Peso Estimado", f"{peso_final:.2f} kg")

elif escolha == "Histórico e Auditoria":
    st.header("📈 Histórico de Pesagens Registradas")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT id, brinco_id, data, peso_estimado, caminho_foto FROM pesagens ORDER BY id DESC", conn)
        conn.close()
        
        if not df.empty:
            # Exibir tabela de dados
            st.dataframe(df[['id', 'brinco_id', 'data', 'peso_estimado']], use_container_width=True)
            
            st.divider()
            st.subheader("🔍 Auditoria de Pesagem Individual")
            
            id_selecionado = st.selectbox("Selecione o ID da pesagem para verificar a foto:", df['id'])
            
            # Recuperar dados da foto selecionada
            dados_foto = df[df['id'] == id_selecionado]
            nome_foto = dados_foto['caminho_foto'].values[0]
            caminho_foto_audit = os.path.join(IMG_SAVE_PATH, nome_foto)
            
            if os.path.exists(caminho_foto_audit):
                st.image(Image.open(caminho_foto_audit), caption=f"Foto original da Pesagem #{id_selecionado}", width=500)
                
                # Botão de Download da Foto
                with open(caminho_foto_audit, "rb") as file:
                    st.download_button(
                        label="📥 Baixar Foto de Auditoria",
                        data=file,
                        file_name=nome_foto,
                        mime="image/jpeg"
                    )
            else:
                st.warning("Arquivo de imagem não encontrado no servidor.")
        else:
            st.info("Ainda não existem pesagens registradas no banco de dados.")
    else:
        st.error("Banco de dados não encontrado.")