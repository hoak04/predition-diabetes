import streamlit as st
import pandas as pd
import joblib
import requests
import os

# Configuração da página deve ser o primeiro comando Streamlit
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")

# Função para carregar arquivos .pkl de URLs
@st.cache_resource
def carregar_modelo_remoto(url, nome_arquivo):
    if not os.path.exists(nome_arquivo):
        r = requests.get(url)
        with open(nome_arquivo, 'wb') as f:
            f.write(r.content)
    return joblib.load(nome_arquivo)

# URLs diretas dos arquivos no Google Drive
url_modelo = "https://drive.google.com/uc?export=download&id=1bnOTS_hydnw6M925PqJCSqVoniQmT_BE"
url_scaler = "https://drive.google.com/uc?export=download&id=14B1EO0nN_L2flEJzZBNESTAjDiXEdJ5O"

# Carregamento do modelo e scaler
modelo = carregar_modelo_remoto(url_modelo, "modelo.pkl")
scaler = carregar_modelo_remoto(url_scaler, "scaler.pkl")

# Lista EXATA das colunas usadas no treinamento
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

# Interface de entrada do usuário
st.title("🩺 Preditor de Diabetes")

# Campos numéricos
entrada = {
    "Age": st.slider("Idade", 1, 120, 45),
    "BMI": st.number_input("IMC", 10.0, 60.0, 28.5),
    "Waist_Circumference": st.number_input("Cintura (cm)", 50.0, 200.0, 90.0),
    "Fasting_Blood_Glucose": st.number_input("Glicose jejum", 50, 300, 100),
    "HbA1c": st.number_input("HbA1c", 3.0, 15.0, 5.8),
    "Blood_Pressure_Systolic": st.number_input("Pressão Sistólica", 80, 200, 120),
    "Blood_Pressure_Diastolic": st.number_input("Pressão Diastólica", 40, 130, 75),
    "Cholesterol_Total": st.number_input("Colesterol Total", 100.0, 300.0, 190.0),
    "Cholesterol_HDL": st.number_input("Colesterol HDL", 20.0, 100.0, 55.0),
    "Cholesterol_LDL": st.number_input("Colesterol LDL", 30.0, 200.0, 110.0),
    "GGT": st.number_input("GGT", 10, 100, 30),
    "Serum_Urate": st.number_input("Ácido Úrico", 1.0, 10.0, 5.2),
    "Dietary_Intake_Calories": st.number_input("Calorias ingeridas", 1000, 5000, 2200),
    "Family_History_of_Diabetes": 1,
    "Previous_Gestational_Diabetes": 0
}

# Simulação de categorias binárias — você pode adaptar com seletores
entrada.update({
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
})

# Construir o DataFrame e reordenar com as colunas corretas
df = pd.DataFrame([entrada])
df = df.reindex(columns=colunas_modelo, fill_value=0)

# Verificação visual
st.subheader("🔎 Dados para predição")
st.write(df)

# Predição
try:
    dados_normalizados = scaler.transform(df)

    if st.button("🔍 Prever"):
        proba = modelo.predict_proba(dados_normalizados)[0]
        prob_diabetes = round(proba[1] * 100, 2)
        prob_normal = round(proba[0] * 100, 2)

        # Resultado da predição
        if prob_diabetes >= 50:
            st.error(f"⚠️ Chance de diabetes: {prob_diabetes}%")
        else:
            st.success(f"🟢 Baixa chance de diabetes ({prob_diabetes}%)")

        st.write(f"🔹 Sem diabetes: {prob_normal}%")
        st.write(f"🔸 Com diabetes: {prob_diabetes}%")

        # Importância das variáveis
        importancias = modelo.feature_importances_
        df_importancia = pd.DataFrame({
            'feature': colunas_modelo,
            'importancia': importancias
        })

        top_features = df_importancia.sort_values(by="importancia", ascending=False).head(5)
        st.subheader("📊 Variáveis mais influentes nesta previsão")
        st.table(top_features)

        # Sugestões com base nas variáveis
        st.subheader("💡 Sugestões para reduzir o risco")
        sugestoes = []

        if entrada["BMI"] > 25:
            sugestoes.append(f"• Reduzir o IMC (atualmente {entrada['BMI']:.1f}) para abaixo de 25.")
        if entrada["Fasting_Blood_Glucose"] > 100:
            sugestoes.append(f"• Reduzir a glicose de jejum (atualmente {entrada['Fasting_Blood_Glucose']}) para < 100 mg/dL.")
        if entrada["HbA1c"] > 5.7:
            sugestoes.append(f"• Reduzir HbA1c (atualmente {entrada['HbA1c']}) para < 5.7.")
        if entrada["Cholesterol_LDL"] > 130:
            sugestoes.append(f"• Reduzir o colesterol LDL (atualmente {entrada['Cholesterol_LDL']}) para < 130 mg/dL.")
        if entrada["Waist_Circumference"] > 102:
            sugestoes.append(f"• Reduzir a circunferência da cintura (atualmente {entrada['Waist_Circumference']} cm) para < 102 cm.")

        if sugestoes:
            for s in sugestoes:
                st.markdown(s)
        else:
            st.markdown("✅ Nenhuma recomendação específica — seus principais indicadores estão dentro de padrões saudáveis.")
except Exception as e:
    st.error(f"Erro na predição: {e}")
