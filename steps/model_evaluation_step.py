# model_evaluation_step.py
import logging
from zenml import step
from src.model_evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
import pandas as pd
from zenml.client import Client
# Get the active experiment tracker from ZenML
experiment_name = Client().active_stack.experiment_tracker
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def model_evaluation_step(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    A ZenML step to evaluate a trained model and log evaluation metrics using ModelEvaluator.

    Parameters:
   
    model : BaseEstimator
        The trained model to evaluate.
    X_test : pd.DataFrame
        The test features.
    y_test : pd.Series
        The true labels for the test set.

    Returns:
    
    dict
        A dictionary containing accuracy, precision, recall, and f1-score.
    """
    logger.info("Starting model evaluation step...")

    #evaluator = ModelEvaluator(model=model, X_test=X_test, y_test=y_test, experiment_name=experiment_name.name)
    evaluator= ModelEvaluator(model=model, X_test=X_test, y_test=y_test)
    # Perform evaluation and log metrics
    results = evaluator.evaluate_model()
    logger.info("Model evaluation completed successfully.")
    
    return results
