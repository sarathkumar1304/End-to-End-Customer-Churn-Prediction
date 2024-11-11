import json

import requests

url = "http://127.0.0.1:8000/invocations"   

# Sample input data for prediction
# Replace the values with the actual features your model expects
input_data = {
    "dataframe_records": [
        {
        'Age':25, 
        'Gender':0, 
        'Tenure':10, 
        'Usage Frequency':5,
        'Support Calls':7, 
        'Payment Delay':3, 
        'Subscription Type':0,
        'Contract Length':1, 
        'Total Spend':23, 
        'Last Interaction':4 
        }
    ]
}
print(input_data)
# Convert the input data to JSON format
json_data = json.dumps(input_data)

# Set the headers for the request
headers = {"Content-Type": "application/json"}

# Send the POST request to the server
response = requests.post(url, headers=headers, data=json_data)

# Check the response status code
if response.status_code == 200:
    # If successful, print the prediction result
    prediction = response.json()
    print("Prediction:", prediction)
else:
    # If there was an error, print the status code and the response
    print(f"Error: {response.status_code}")
    print(response.text)
    
    