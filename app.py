import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💓 Heart Disease Prediction App")

st.write("Enter the following medical information to predict your heart disease risk:")

# User input form
age = st.number_input("Age", min_value=18, max_value=100, value=45)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
chol = st.number_input("Cholesterol Level", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])

# Predict button
if st.button("Predict"):

    # Encode categorical inputs
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # One-hot encode manually (simplified)
    cp_ATA = 1 if cp == "ATA" else 0
    cp_NAP = 1 if cp == "NAP" else 0
    cp_ASY = 1 if cp == "ASY" else 0

    restecg_ST = 1 if restecg == "ST" else 0
    restecg_LVH = 1 if restecg == "LVH" else 0

    # Combine all features in the correct order
    input_data = np.array([[age, sex, cp_ATA, cp_NAP, cp_ASY, chol,
                            fbs, restecg_ST, restecg_LVH, thalach, exang]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 1:
        st.error("⚠️ The model predicts that you may be at risk of heart disease.")
    else:
        st.success("💚 The model predicts that you are unlikely to have heart disease.")
