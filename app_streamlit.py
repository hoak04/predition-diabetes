import streamlit as st
import pandas as pd
import joblib
import requests
import os

st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")
# Função para baixar e carregar arquivos .pkl
@st.cache_resource
def carregar_modelo_remoto(url, nome_arquivo):
    if not os.path.exists(nome_arquivo):
        r = requests.get(url)
        with open(nome_arquivo, 'wb') as f:
            f.write(r.content)
    return joblib.load(nome_arquivo)

# URLs diretas do Google Drive
url_modelo = "https://drive.google.com/uc?export=download&id=1bnOTS_hydnw6M925PqJCSqVoniQmT_BE"
url_scaler = "https://drive.google.com/uc?export=download&id=14B1EO0nN_L2flEJzZBNESTAjDiXEdJ5O"

# Carregar modelo e scaler
modelo = carregar_modelo_remoto(url_modelo, "modelo.pkl")
scaler = carregar_modelo_remoto(url_scaler, "scaler.pkl")

# Interface Streamlit


# Inputs
age = st.slider("Idade", 1, 120, 45)
bmi = st.number_input("IMC", 10.0, 60.0, 28.5)
waist = st.number_input("Cintura (cm)", 50.0, 200.0, 90.0)
glucose = st.number_input("Glicose jejum", 50, 300, 100)
hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.8)
hdl = st.number_input("Colesterol HDL", 20.0, 100.0, 55.0)
ldl = st.number_input("Colesterol LDL", 30.0, 200.0, 110.0)
chol_total = st.number_input("Colesterol Total", 100.0, 300.0, 190.0)
ggt = st.number_input("GGT", 10, 100, 30)
urate = st.number_input("Ácido Úrico (Serum Urate)", 1.0, 10.0, 5.2)
calories = st.number_input("Calorias ingeridas", 1000, 5000, 2200)
bp_sys = st.number_input("Pressão Sistólica", 80, 200, 120)
bp_dia = st.number_input("Pressão Diastólica", 40, 130, 75)

# Exemplo de entrada com colunas binárias simuladas
entrada = {
    "Age": age,
    "BMI": bmi,
    "Waist_Circumference": waist,
    "Fasting_Blood_Glucose": glucose,
    "Sex_Male": 1,
    "Alcohol_Consumption_None": 1,
    "Alcohol_Consumption_Moderate": 0,
    "Smoking_Status_Never": 1,
    "Smoking_Status_Former": 0,
    "Physical_Activity_Level_Moderate": 1,
    "Physical_Activity_Level_Low": 0,
    "Family_History_of_Diabetes": 1,
    "Previous_Gestational_Diabetes": 0,
    "Cholesterol_Total": chol_total,
    "Cholesterol_HDL": hdl,
    "Cholesterol_LDL": ldl,
    "HbA1c": hba1c,
    "GGT": ggt,
    "Serum_Urate": urate,
    "Dietary_Intake_Calories": calories,
    "Ethnicity_White": 1,
    "Ethnicity_Black": 0,
    "Ethnicity_Hispanic": 0,
    "Blood_Pressure_Diastolic": bp_dia,
    "Blood_Pressure_Systolic": bp_sys
}

# Converter para DataFrame
df = pd.DataFrame([entrada])

# Verificações
st.subheader("🔎 Verificação")
st.write("Colunas enviadas:", df.columns.tolist())
st.write("Shape:", df.shape)

# Predição
try:
    dados_normalizados = scaler.transform(df)
    if st.button("🔍 Prever"):
        pred = modelo.predict(dados_normalizados)[0]
        st.success("✅ Diabetes detectado!" if pred == 1 else "🟢 Sem sinais de diabetes.")
except Exception as e:
    st.error(f"Erro na predição: {e}")
