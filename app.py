import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Load model and encoders ──
model = joblib.load('loan_model_small.pkl')
encoders = joblib.load('encoders.pkl')

# ── Page config ──
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

st.title("🏦 Loan Approval Predictor")
st.write("Fill in the applicant details below to predict loan default risk.")

# ── Input form ──
st.subheader("👤 Personal Details")
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)

with col2:
    person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    cb_person_default_on_file = st.selectbox("Previous Default on File?", ['Y', 'N'])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)

st.subheader("💰 Loan Details")
col3, col4 = st.columns(2)

with col3:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=100000, value=10000, step=500)
    loan_int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=25.0, value=10.0, step=0.1)
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

with col4:
    loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 
                                                'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0
    st.metric("Loan % of Income", f"{loan_percent_income:.1%}")

# ── Predict button ──
if st.button("🔍 Predict", use_container_width=True):

    # encode categorical inputs
    home_enc    = encoders['person_home_ownership'].transform([person_home_ownership])[0]
    intent_enc  = encoders['loan_intent'].transform([loan_intent])[0]
    grade_enc   = encoders['loan_grade'].transform([loan_grade])[0]
    default_enc = encoders['cb_person_default_on_file'].transform([cb_person_default_on_file])[0]

    # build input row
    input_data = pd.DataFrame([[
        0,                    # id
        person_age,           # person_age
        person_income,        # person_income
        home_enc,             # person_home_ownership
        person_emp_length,    # person_emp_length
        intent_enc,           # loan_intent
        grade_enc,            # loan_grade
        loan_amnt,            # loan_amnt
        loan_int_rate,        # loan_int_rate
        loan_percent_income,  # loan_percent_income
        default_enc,          # cb_person_default_on_file
        cb_person_cred_hist_length  # cb_person_cred_hist_length
    ]], columns=[
        'id', 'person_age', 'person_income', 'person_home_ownership',
        'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
        'cb_person_cred_hist_length'
    ])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 0:
        st.success(f"✅ APPROVED — Low default risk ({probability[0]:.1%} confidence)")
    else:
        st.error(f"❌ HIGH RISK — Likely to default ({probability[1]:.1%} confidence)")

    # confidence bar
    st.write("**Confidence Breakdown**")
    col5, col6 = st.columns(2)
    col5.metric("✅ Will Repay", f"{probability[0]:.1%}")
    col6.metric("❌ Will Default", f"{probability[1]:.1%}")

    st.subheader("📊 Key Factors Influencing Decision")

    if person_income < 40000:
        st.write("• Lower income slightly increases default risk")
    if cb_person_default_on_file == 'Y':
        st.write("• Previous default significantly increases risk")
    if cb_person_cred_hist_length < 3:
        st.write("• Short credit history reduces reliability")

