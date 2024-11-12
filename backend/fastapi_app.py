from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd

app = FastAPI()

# Define the structure of the incoming data
class InputData(BaseModel):
    dataframe_records: list[dict]  # List of dictionaries (like rows in a DataFrame)

# Define endpoint to receive data and make a prediction
@app.post("/predict")
async def make_prediction(input_data: InputData):
    # URL for MLflow's prediction server
    mlflow_url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}

    # Prepare the JSON data to send to MLflow
    json_data = {
        "dataframe_records": input_data.dataframe_records
    }

    try:
        # Send data to MLflow and get prediction
        response = requests.post(mlflow_url, headers=headers, json=json_data)
        response.raise_for_status()  # Raise an error for a failed request
        return response.json()  # Return MLflow's prediction result

    except requests.exceptions.HTTPError as err:
        raise HTTPException(status_code=response.status_code, detail=str(err))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
