import json
import numpy as np 
import pandas as pd 
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step
def predictor(service: MLFlowDeploymentService, input_data: str) -> np.ndarray:
    """
    Makes predictions using a deployed MLflow model service.

    Args:
    service (MLFlowDeploymentService): The MLflow deployment service to use for prediction.
    input_data (str): A JSON string containing the input data for prediction.

    Returns:
    np.ndarray: The predicted output.
    """
    service.start(timeout=60)
    
    # Load input data as a JSON object
    data = json.loads(input_data)  # Use json.loads to parse the input string

    # Remove unnecessary keys if they exist
    data.pop("Columns", None)
    data.pop("index", None)

    expected_columns = [
        'Age', 
        'Gender', 
        'Tenure', 
        'Usage Frequency',
        'Support Calls', 
        'Payment Delay', 
        'Subscription Type',
        'Contract Length', 
        'Total Spend', 
        'Last Interaction'
    ]

    # Create a DataFrame from the provided data
    df = pd.DataFrame(data['data'], columns=expected_columns)

    # Convert DataFrame to the appropriate format for prediction
    json_list = list(df.T.to_dict().values())  # Convert DataFrame to a list of dictionaries
    data_array = np.array(json_list)  # Convert list of dictionaries to a numpy array
    
    # Make predictions using the deployed service
    prediction = service.predict(data_array)
    
    return prediction
#http://127.0.0.1:8000/invocations