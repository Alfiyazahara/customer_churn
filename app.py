import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Judul Aplikasi
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📱 Customer Churn Prediction")
st.markdown("Masukkan data pelanggan di bawah ini untuk memprediksi potensi Churn.")

# 2. Fungsi untuk Memuat Artifact
@st.cache_resource
def load_artifacts():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('best_model_churn.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

try:
    label_encoders, standard_scaler, model = load_artifacts()

    # 3. Form Input Pengguna
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", options=['Male', 'Female'])
            tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
            usage_freq = st.number_input("Usage Frequency", min_value=0, value=10)
            support_calls = st.number_input("Support Calls", min_value=0, value=2)

        with col2:
            payment_delay = st.number_input("Payment Delay (Days)", min_value=0, value=5)
            subscription = st.selectbox("Subscription Type", options=['Basic', 'Standard', 'Premium'])
            contract = st.selectbox("Contract Length", options=['Monthly', 'Quarterly', 'Annual'])

        submit = st.form_submit_button("Predict Churn")

    # 4. Logika Prediksi
    if submit:
        # Buat dataframe dari input
        input_df = pd.DataFrame({
            'Gender': [gender],
            'Tenure': [tenure],
            'Usage Frequency': [usage_freq],
            'Support Calls': [support_calls],
            'Payment Delay': [payment_delay],
            'Subscription Type': [subscription],
            'Contract Length': [contract]
        })

        # Preprocessing: Label Encoding
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        # Preprocessing: Scaling
        input_scaled = standard_scaler.transform(input_df)

        # Prediksi
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Tampilkan Hasil
        st.divider()
        if prediction == 1:
            st.error(f"Hasil Prediksi: **CHURN** (Probabilitas: {probability:.2%})")
            st.warning("Pelanggan ini berisiko tinggi untuk berhenti berlangganan.")
        else:
            st.success(f"Hasil Prediksi: **NOT CHURN** (Probabilitas Churn: {probability:.2%})")
            st.info("Pelanggan ini cenderung akan tetap berlangganan.")

except FileNotFoundError:
    st.error("File model (.pkl) tidak ditemukan. Pastikan Anda sudah menjalankan tahap export model.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
