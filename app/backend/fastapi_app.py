# backend/fastapi_app.py
from fastapi import FastAPI, HTTPException
import requests
import json

app = FastAPI()

@app.post("/predict")
async def predict(input_data: dict):
    url = "http://127.0.0.1:8000/invocations"  # MLflow prediction endpoint
    headers = {"Content-Type": "application/json"}
    
    # Prepare data for MLflow in the required format
    json_data = json.dumps({"dataframe_records": [input_data]})
    
    # Send the POST request to MLflow
    response = requests.post(url, headers=headers, data=json_data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# Run this app: uvicorn fastapi_app:app --host 127.0.0.1 --port 8001

