import streamlit as st
import pandas as pd
from make_prediction import make_prediction, predict_from_mlflow
import requests

def project_ui():
    
    st.write("Enter Customer Details")

    # Collect user inputs
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    gender_encoded = 1 if gender == "Male" else 0

    tenure = st.number_input("Tenure (months)", min_value=0, step=1)
    usage_frequency = st.number_input("Usage Frequency", min_value=0, step=1)
    support_calls = st.number_input("Support Calls", min_value=0, step=1)
    payment_delay = st.number_input("Payment Delay", min_value=0, step=1)

    subscription_type = st.selectbox("Subscription Type", options=["Standard", "Basic", "Premium"])
    subscription_type_encoded = {"Standard": 2, "Basic": 0, "Premium": 1}[subscription_type]

    contract_length = st.selectbox("Contract Length", options=["Annual", "Monthly", "Quarterly"])
    contract_length_encoded = {"Annual": 0, "Monthly": 1, "Quarterly": 2}[contract_length]

    total_spend = st.number_input("Total Spend", min_value=0.0, step=1.0)
    last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, step=1)

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type_encoded],
        "Contract Length": [contract_length_encoded],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction],
    })
    st

    if st.button("Predict Churn"):
    # Make prediction using the model
        prediction = make_prediction(input_data)
    
        if prediction is not None:
            # st.write(prediction)  # Optional: Display the raw prediction output
            
            # Check if the prediction is in the expected format
            if isinstance(prediction, dict) and 'predictions' in prediction:
                if len(prediction['predictions']) > 0:
                    # Extract the first prediction value and convert it to an integer
                    churn_value = int(prediction['predictions'][0])  # Converts float to int
                    
                    # Determine the churn prediction
                    churn_prediction = "Will Churn" if churn_value == 1 else "Won't Churn"
                    
                    # Display the churn prediction result
                    st.success(churn_prediction)
                else:
                    st.error("No predictions found.")
            else:
                st.error("Prediction result format is unexpected.")
        else:
            st.error("Prediction request failed.")




