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
# O caminho abaixo busca a pasta 'models' um nível acima da pasta 'src'
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_peso_bois.h5')

# Garantir que a pasta de fotos existe
if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)

# --- FUNÇÕES DE PROCESSAMENTO VISUAL AVANÇADO (RAYVORA VISION) ---

def aplicar_clahe(imagem_np):
    """
    Equalização de Histograma Adaptativa Limitada por Contraste.
    Melhora a definição de contornos do animal em condições de sombra ou sol forte.
    """
    # Converter para escala de cinza (exigência do CLAHE)
    gray = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2GRAY)
    
    # Criar objeto CLAHE: clipLimit=2.0 (evita ruído), tileGridSize=(8,8) (análise regional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_equalizada = clahe.apply(gray)
    
    # Retornar em RGB para compatibilidade com a rede neural
    return cv2.cvtColor(img_equalizada, cv2.COLOR_GRAY2RGB)

def preparar_imagem_ia(img_pil):
    """Pipeline completo de tratamento de dados antes da inferência"""
    img_np = np.array(img_pil)
    
    # 1. Tratamento de iluminação (Foco Comercial)
    img_tratada = aplicar_clahe(img_np)
    
    # 2. Redimensionamento para o padrão de treino (128x128)
    img_res = cv2.resize(img_tratada, (128, 128))
    
    # 3. Normalização de pixels (0 a 1)
    img_final = img_res / 255.0
    
    # Expandir dimensão para (1, 128, 128, 3)
    return np.expand_dims(img_final, axis=0), img_tratada

# --- FUNÇÕES DE LÓGICA DE NEGÓCIO ---

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
        st.error(f"Erro Crítico: Modelo não encontrado em {MODEL_PATH}")
        return None

def calcular_gmd(brinco_id, peso_atual):
    """Calcula Ganho Médio Diário comparando com o histórico do brinco"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT data, peso_estimado FROM pesagens WHERE brinco_id = ? ORDER BY id DESC LIMIT 1"
    df_ant = pd.read_sql_query(query, conn, params=(brinco_id,))
    conn.close()

    if not df_ant.empty:
        peso_ant = df_ant['peso_estimado'].values[0]
        data_ant = datetime.strptime(df_ant['data'].values[0], "%d/%m/%Y %H:%M:%S")
        hoje = datetime.now()
        
        dias = (hoje - data_ant).days
        if dias <= 0: dias = 1 # Evita erro matemático se pesado no mesmo dia
        
        gmd = (peso_atual - peso_ant) / dias
        return gmd, dias, peso_ant
    return None, None, None

# --- INTERFACE DO USUÁRIO (UI) ---

st.set_page_config(page_title="Rayvora Vision v2.0", layout="wide", page_icon="🐄")
init_db()

st.title("🐄 Rayvora Vision: Inteligência Bovina")
st.markdown("Sistema de pesagem visual com correção de iluminação dinâmica.")

menu = ["🏠 Início & Pesagem", "📊 Histórico & Analytics"]
escolha = st.sidebar.selectbox("Navegação Principal", menu)

if escolha == "🏠 Início & Pesagem":
    st.header("⚖️ Nova Estimativa de Peso")
    
    col_ctrl, col_orig, col_ia = st.columns([1, 1, 1])
    
    with col_ctrl:
        brinco = st.text_input("Identificação (Brinco):", placeholder="Ex: BOI_001")
        foto = st.file_uploader("Carregar Imagem (Vista Traseira)", type=['jpg', 'jpeg', 'png'])
        
    if foto is not None:
        img_original = Image.open(foto).convert('RGB')
        
        with col_orig:
            st.image(img_original, caption="Imagem Original (Campo)", use_container_width=True)
        
        if st.button("🚀 Processar Pesagem Inteligente"):
            with st.spinner('Otimizando imagem e consultando IA...'):
                # Pipeline de Visão
                img_pronta, img_viz = preparar_imagem_ia(img_original)
                
                # Exibir para auditoria o que a IA está analisando
                with col_ia:
                    st.image(img_viz, caption="Filtro CLAHE (Detecção de Contorno)", use_container_width=True)
                
                # Inferência
                model = load_model_ia()
                if model:
                    peso_final = float(model.predict(img_pronta)[0][0])
                    
                    # Filtro de Validação de Peso
                    if peso_final < 50 or peso_final > 1500:
                        st.error(f"❌ Erro de Validação: Peso estimado ({peso_final:.2f}kg) fora da faixa bovina.")
                    else:
                        # Cálculo GMD
                        gmd, dias, peso_ant = calcular_gmd(brinco, peso_final)
                        
                        # Persistência
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

                        # Feedback Visual
                        st.balloons()
                        st.success(f"✅ Pesagem registrada: {peso_final:.2f} kg")
                        
                        if gmd is not None:
                            m1, m2 = st.columns(2)
                            m1.metric("GMD (Ganho Médio Diário)", f"{gmd:.3f} kg/dia", delta=f"{peso_final - peso_ant:.2f} kg")
                            m2.info(f"Intervalo desde o último manejo: {dias} dias.")

elif escolha == "📊 Histórico & Analytics":
    st.header("📈 Dashboard de Desempenho do Rebanho")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM pesagens", conn)
        conn.close()

        if not df.empty:
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            
            # Filtro por animal específico
            lista_bois = df['brinco_id'].unique()
            boi_sel = st.selectbox("Selecione o Animal para Auditoria:", lista_bois)
            
            df_boi = df[df['brinco_id'] == boi_sel].sort_values('data')
            
            # Gráfico Interativo Plotly
            fig = px.line(df_boi, x='data', y='peso_estimado', 
                          title=f'Histórico de Ganho de Peso: {boi_sel}',
                          markers=True, line_shape='spline',
                          color_discrete_sequence=['#2ecc71'])
            fig.update_layout(yaxis_title="Peso (kg)", xaxis_title="Data do Manejo")
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas Totais
            k1, k2, k3 = st.columns(3)
            k1.metric("Peso na Primeira Pesagem", f"{df_boi['peso_estimado'].iloc[0]:.2f} kg")
            k2.metric("Peso Atual (Última)", f"{df_boi['peso_estimado'].iloc[-1]:.2f} kg")
            ganho_total = df_boi['peso_estimado'].iloc[-1] - df_boi['peso_estimado'].iloc[0]
            k3.metric("Ganho Acumulado no Ciclo", f"{ganho_total:.2f} kg")

            st.divider()
            st.subheader("📁 Gerenciamento de Dados")
            c_db, c_csv = st.columns(2)
            with open(DB_PATH, "rb") as f:
                c_db.download_button("📥 Baixar Base de Dados (.db)", f, "rayvora_database.db")
            
            csv = df_boi.to_csv(index=False).encode('utf-8')
            c_csv.download_button("📥 Baixar Histórico deste Animal (.csv)", csv, f"historico_{boi_sel}.csv", "text/csv")
        else:
            st.info("O banco de dados ainda está vazio. Realize uma pesagem para iniciar o dashboard.")
    else:
        st.error("Banco de dados não localizado no servidor.")