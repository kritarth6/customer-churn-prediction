import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# 🎨 TITLE
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# SIDEBAR INPUTS
st.sidebar.header("📋 Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total_charges = st.sidebar.slider("Total Charges", 0, 10000, 2000)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Encode inputs
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract_map[contract]],
    "InternetService": [internet_map[internet]]
})

# 🎯 PREDICTION BUTTON
if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)

    # RESULT
    with col1:
        if prediction == 1:
            st.error(f"⚠️ Customer Likely to Churn\nProbability: {prob:.2f}")
        else:
            st.success(f"✅ Customer Not Likely to Churn\nProbability: {prob:.2f}")

    # 📊 GAUGE CHART
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    # 📈 FEATURE IMPORTANCE (if model supports)
    st.subheader("📊 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = input_data.columns

        fig2 = px.bar(x=features, y=importance, title="Feature Importance")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Model does not support feature importance")

# 📂 CSV UPLOAD FEATURE
st.subheader("📂 Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    preds = model.predict(data)

    data["Prediction"] = preds

    st.dataframe(data.head())

    st.download_button(
        label="Download Results",
        data=data.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )