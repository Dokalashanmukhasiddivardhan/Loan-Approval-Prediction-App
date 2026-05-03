import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.set_page_config(page_title="Loan Prediction", layout="centered")
st.title(" Loan Approval Prediction App")

st.markdown("### Enter Applicant Details")

# -------------------------------
# INPUT UI (Professional)
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    income = st.number_input("Income", min_value=0, value=30000)
    experience = st.slider("Experience (Years)", 0, 20, 2)
    home = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
    intent = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Business"])
    interest = st.slider("Interest Rate (%)", 5, 25, 10)
    percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.3)
    credit_history = st.slider("Credit History Length", 0, 15, 3)
    credit_score = st.slider("Credit Score", 300, 900, 650)
    default = st.selectbox("Previous Default", ["No", "Yes"])

# -------------------------------
# ENCODING INPUTS
# -------------------------------

gender = 1 if gender == "Male" else 0
education = 1 if education == "Graduate" else 0
home = {"Rent": 0, "Own": 1, "Mortgage": 2}[home]
intent = {"Personal": 0, "Education": 1, "Medical": 2, "Business": 3}[intent]
default = 1 if default == "Yes" else 0

# Input array
input_data = np.array([
    age, gender, education, income, experience,
    home, loan_amount, intent, interest,
    percent_income, credit_history, credit_score, default
]).reshape(1, -1)

# Scale
input_data = scaler.transform(input_data)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button(" Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Loan Approved (Confidence: {probability:.2f})")
    else:
        st.error(f" Loan Rejected (Confidence: {1 - probability:.2f})")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built using Machine Learning + Streamlit ")
