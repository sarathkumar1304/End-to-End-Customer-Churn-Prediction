import os
import logging
from typing import Annotated
import mlflow
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Import ModelBuilding class
from src.model_building import ModelBuilding

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

# Define model metadata
model_metadata = Model(
    name="customer_churn_prediction",
    version=None,
    license="Apache-2.0",
    description="Customer churn prediction model for Telecom company.",
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # file:///home/sarath_kumar/.config/zenml/local_stores/b878ca30-c25c-4712-9a5a-a299384dcb87/mlruns/649008275814095771/a4672e3f3d6840cd8f5114939de29272/artifacts/model/model.pkl
#
# Adjusted model_builder_step function
@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model_metadata)
def model_builder_step(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[
    Pipeline,ArtifactConfig(name = "sklearn_pipeline",is_model_artifact = True)]:
    """
    ZenML step to create, preprocess, train, and return a specified model.

    Parameters
    
    model_name : str
        Name of the model to create.
    X_train : pd.DataFrame
        Training data features.
    y_train : pd.Series
        Training data labels/target.

    Returns
    
    Any
        The trained model or pipeline including preprocessing.

    """

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

    logger.info(f"Categorical columns: {categorical_cols.tolist()}")
    logger.info(f"Numerical columns: {numerical_cols.tolist()}")
    logger.info("Starting model building step...")
    
    if not mlflow.active_run():
        mlflow.start_run()
    
    # Initialize the ModelBuilding class and select model by name
    model_builder = ModelBuilding()
    
    try:
        mlflow.sklearn.autolog()
        model = model_builder.get_model(model_name, X_train, y_train)
        logger.info(f"Model '{model_name}' has been successfully created.")
        # Define the pipeline including the model (assuming no preprocessing here)
        pipeline = Pipeline(steps=[("model", model)])
        # Train the model
        pipeline.fit(X_train, y_train)
        logger.info("Model training completed")
    except ValueError as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        # end the mlflow run
        mlflow.end_run()
    
    return pipeline

    

    

   
        

    # # Save the model pipeline locally after evaluation
    # model_dir = "models"
    # os.makedirs(model_dir, exist_ok=True)  # Ensure the models directory exists
    # model_path = os.path.join(model_dir, "model.pkl")
    # joblib.dump(pipeline, model_path)  # Save model pipeline as 'model.pkl'
    # logger.info(f"Model saved at {model_path}")


# zenml stack register mlflow_stack_customer_churn_prediction -a default -o default -d mlflow -e mlflow_tracker_customer_churn_prediction --set