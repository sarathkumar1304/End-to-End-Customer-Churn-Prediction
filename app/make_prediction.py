
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

