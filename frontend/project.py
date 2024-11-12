import streamlit as st
import pandas as pd
from make_prediction import get_prediction

def project_ui():
    st.title("Customer Churn Prediction")

    
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

    # Create DataFrame of input data for the prediction
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

    
    if st.button("Predict Churn"):
        prediction = get_prediction(input_data)
        
        
        if prediction is not None:
            churn_value = int(prediction['predictions'][0]) 
            churn_prediction = "Will Churn" if churn_value == 1 else "Won't Churn"
            st.success(f"Prediction: {churn_prediction}")
        else:
            st.error("Prediction request failed.")


if __name__ == "__main__":
    project_ui()



