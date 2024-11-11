
import requests
import json
import streamlit as st
import pandas as pd

import json
import requests

def make_prediction(input_data):
    url = "http://127.0.0.1:8000/invocations"  
    headers = {"Content-Type": "application/json"}
    
   
    json_data = {
        "dataframe_records": input_data.to_dict(orient="records")
    }

    try:
        response = requests.post(url, headers=headers, json=json_data)  
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"Request failed: {err}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def predict_from_mlflow(data):
    # URL for the MLflow model server
    mlflow_url = "http://127.0.0.1:8000/invocations"  # Replace with your MLflow server URL

    # Convert DataFrame to JSON format with the correct field for MLflow 2.0
    payload = {
        "dataframe_split": data.to_dict(orient="split")
    }

    # Make the POST request to the MLflow server
    headers = {"Content-Type": "application/json"}
    response = requests.post(mlflow_url, headers=headers, json=payload)

    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None