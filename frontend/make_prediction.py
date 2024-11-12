import requests
import streamlit as st


# Helper function to send data to FastAPI for prediction
def get_prediction(input_data):
    """
    Sends the input data to the FastAPI backend to get a prediction.

    Args:
        input_data (pd.DataFrame): Input data to send to the FastAPI backend

    Returns:
        dict or None: The JSON response from the FastAPI backend, or None if the request failed
    """


    url = "http://127.0.0.1:8001/predict"  # URL of the FastAPI backend
    headers = {"Content-Type": "application/json"}

    json_data = {
        "dataframe_records": input_data.to_dict(orient="records")
    }

    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        return response.json()  # Return JSON response from FastAPI
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
