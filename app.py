import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer

model, scaler, explainer = load_model()

# Suggestions mapping
SUGGESTIONS = {
    'Contract': "💡 Offer a discounted annual or two-year contract to lock in the customer.",
    'MonthlyCharges': "💡 Consider offering a loyalty discount or a cheaper plan bundle.",
    'tenure': "💡 This is a new customer — assign a dedicated onboarding support agent.",
    'OnlineSecurity': "💡 Offer a free trial of Online Security add-on for 3 months.",
    'TechSupport': "💡 Offer free Tech Support upgrade for the next billing cycle.",
    'OnlineBackup': "💡 Highlight the value of Online Backup — offer it at a reduced rate.",
    'InternetService': "💡 Check if the customer is satisfied with internet speed and reliability.",
    'PaymentMethod': "💡 Encourage auto-payment setup with a small monthly discount.",
    'PaperlessBilling': "💡 Educate customer on paperless billing benefits.",
    'TotalCharges': "💡 Review if total spend justifies a loyalty reward or upgrade offer.",
}

# Title
st.title("📉 Customer Churn Predictor")
st.markdown("Enter customer details below to predict whether they are likely to churn.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

st.divider()

# Predict button
if st.button("🔍 Predict Churn", use_container_width=True):

    input_dict = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': {"Yes": 2, "No": 0, "No phone service": 1}[multiple_lines],
        'InternetService': {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
        'OnlineSecurity': {"Yes": 2, "No": 0, "No internet service": 1}[online_security],
        'OnlineBackup': {"Yes": 2, "No": 0, "No internet service": 1}[online_backup],
        'DeviceProtection': {"Yes": 2, "No": 0, "No internet service": 1}[device_protection],
        'TechSupport': {"Yes": 2, "No": 0, "No internet service": 1}[tech_support],
        'StreamingTV': {"Yes": 2, "No": 0, "No internet service": 1}[streaming_tv],
        'StreamingMovies': {"Yes": 2, "No": 0, "No internet service": 1}[streaming_movies],
        'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod': {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3
        }[payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ This customer is likely to **CHURN**")
    else:
        st.success("✅ This customer is likely to **STAY**")

    st.metric("Churn Probability", f"{probability * 100:.1f}%")

    if probability >= 0.7:
        st.warning("🔴 High Risk — Immediate retention action recommended")
    elif probability >= 0.4:
        st.warning("🟡 Medium Risk — Monitor this customer closely")
    else:
        st.info("🟢 Low Risk — Customer appears stable")

    st.divider()

    # SHAP explanation
    shap_values = explainer.shap_values(input_scaled)
    feature_names = list(input_dict.keys())
    shap_vals = shap_values[0, :, 1] if len(np.array(shap_values).shape) == 3 else shap_values[1][0]

    # Top 3 churn reasons
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP': shap_vals
    }).sort_values('SHAP', ascending=False)

    top_churn_drivers = shap_df[shap_df['SHAP'] > 0].head(3)

    if not top_churn_drivers.empty and prediction == 1:
        st.subheader("🔎 Top Reasons for Churn Risk")
        for _, row in top_churn_drivers.iterrows():
            st.markdown(f"- **{row['Feature']}** is pushing this customer towards churn")

        st.divider()

        # Suggestions
        st.subheader("📋 Retention Suggestions")
        shown = 0
        for _, row in top_churn_drivers.iterrows():
            feature = row['Feature']
            if feature in SUGGESTIONS:
                st.markdown(SUGGESTIONS[feature])
                shown += 1
        if shown == 0:
            st.markdown("💡 Consider reaching out personally to understand customer concerns.")

    elif prediction == 0:
        st.subheader("🔎 Why This Customer Is Stable")
        top_stable = shap_df[shap_df['SHAP'] < 0].tail(3)
        for _, row in top_stable.iterrows():
            st.markdown(f"- **{row['Feature']}** is keeping this customer loyal")