# backend/flask_app.py
from flask import Flask, request, jsonify
import requests
import pandas as pd
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()  
    fastapi_url = "http://127.0.0.1:8001/predict"  

    try:
        
        response = requests.post(fastapi_url, json=input_data)
        if response.status_code == 200:
            return jsonify(response.json()) 
        else:
            return jsonify({"error": response.text}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)