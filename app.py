import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure (months)",0,72)
monthly = st.slider("Monthly Charges",0,150)

# create dataframe with all required features
input_data = pd.DataFrame(columns=model.feature_names_in_)

input_data.loc[0] = 0

# fill only the fields we use
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly

if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")
