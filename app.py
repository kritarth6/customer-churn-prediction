import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")
st.write(
    "Predict whether a telecom customer is likely to **leave the service (churn)** based on customer information."
)

st.divider()

# Create columns
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)

st.divider()

# Prepare dataframe
input_data = pd.DataFrame(columns=model.feature_names_in_)
input_data.loc[0] = 0

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges

# Prediction button
if st.button("🔍 Predict Churn", use_container_width=True):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.write(f"Churn Probability: **{probability:.2%}**")

st.divider()

st.caption("Machine Learning Model: Random Forest | Built with Streamlit")