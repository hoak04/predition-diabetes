
import streamlit as st
import pandas as pd
import joblib
import requests
import os
import shap
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")

usuarios = {"admin": "1234"}
if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:
    st.title("🔐 Login")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if usuario in usuarios and usuarios[usuario] == senha:
            st.session_state.logado = True
            st.rerun()
        else:
            st.error("❌ Usuário ou senha incorretos.")
    st.stop()

url_modelo = "https://drive.google.com/uc?export=download&id=1bnOTS_hydnw6M925PqJCSqVoniQmT_BE"
url_scaler = "https://drive.google.com/uc?export=download&id=14B1EO0nN_L2flEJzZBNESTAjDiXEdJ5O"

@st.cache_resource
def carregar_modelo(url, nome):
    if not os.path.exists(nome):
        r = requests.get(url)
        with open(nome, 'wb') as f:
            f.write(r.content)
    return joblib.load(nome)

modelo = carregar_modelo(url_modelo, "modelo.pkl")
scaler = carregar_modelo(url_scaler, "scaler.pkl")

colunas_modelo = [
    "Age", "BMI", "Waist_Circumference", "Fasting_Blood_Glucose", "HbA1c",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Cholesterol_Total",
    "Cholesterol_HDL", "Cholesterol_LDL", "GGT", "Serum_Urate", "Dietary_Intake_Calories",
    "Family_History_of_Diabetes", "Previous_Gestational_Diabetes",
    "Sex_Male_Female", "Sex_Male_Male", "Ethnicity_Asian", "Ethnicity_Black",
    "Ethnicity_Hispanic", "Ethnicity_White", "Physical_Activity_Level_High",
    "Physical_Activity_Level_Low", "Physical_Activity_Level_Moderate",
    "Alcohol_Consumption_Heavy", "Alcohol_Consumption_Moderate",
    "Smoking_Status_Current", "Smoking_Status_Former", "Smoking_Status_Never"
]

st.title("🩺 Preditor de Diabetes com SHAP")

entrada = {
    "Age": st.slider("Idade", 1, 120, 45),
    "BMI": st.number_input("IMC", 10.0, 60.0, 28.5),
    "Waist_Circumference": st.number_input("Cintura (cm)", 50.0, 200.0, 90.0),
    "Fasting_Blood_Glucose": st.number_input("Glicose jejum", 50, 300, 100),
    "HbA1c": st.number_input("HbA1c", 3.0, 15.0, 5.8),
    "Blood_Pressure_Systolic": st.number_input("Pressão Sistólica", 80, 200, 120),
    "Blood_Pressure_Diastolic": st.number_input("Pressão Diastólica", 40, 130, 75),
    "Cholesterol_Total": st.number_input("Colesterol Total", 100.0, 300.0, 190.0),
    "Cholesterol_HDL": st.number_input("HDL", 20.0, 100.0, 55.0),
    "Cholesterol_LDL": st.number_input("LDL", 30.0, 200.0, 110.0),
    "GGT": st.number_input("GGT", 10, 100, 30),
    "Serum_Urate": st.number_input("Ácido Úrico", 1.0, 10.0, 5.2),
    "Dietary_Intake_Calories": st.number_input("Calorias", 1000, 5000, 2200),
    "Family_History_of_Diabetes": 1,
    "Previous_Gestational_Diabetes": 0,
    "Sex_Male_Female": 0,
    "Sex_Male_Male": 1,
    "Ethnicity_Asian": 0,
    "Ethnicity_Black": 0,
    "Ethnicity_Hispanic": 0,
    "Ethnicity_White": 1,
    "Physical_Activity_Level_High": 0,
    "Physical_Activity_Level_Low": 0,
    "Physical_Activity_Level_Moderate": 1,
    "Alcohol_Consumption_Heavy": 0,
    "Alcohol_Consumption_Moderate": 1,
    "Smoking_Status_Current": 0,
    "Smoking_Status_Former": 0,
    "Smoking_Status_Never": 1
}

df = pd.DataFrame([entrada])
df = df.reindex(columns=colunas_modelo, fill_value=0)

try:
    dados_normalizados = scaler.transform(df)
    if st.button("🔍 Prever"):
        proba = modelo.predict_proba(dados_normalizados)[0][1]
        st.success(f"Risco de diabetes: {proba * 100:.2f}%")

        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(pd.DataFrame(df, columns=colunas_modelo))

        st.subheader("📊 SHAP - explicação da predição")
        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            df.iloc[0],
            max_display=10,
            show=False
        )
        st.pyplot(fig)

except Exception as e:
    st.error(f"Erro na predição: {e}")
