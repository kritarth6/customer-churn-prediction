import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure (months)",0,72)
monthly = st.slider("Monthly Charges",0,150)

input_data = pd.DataFrame(
    [[tenure, monthly]],
    columns=["tenure","MonthlyCharges"]
)

if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")