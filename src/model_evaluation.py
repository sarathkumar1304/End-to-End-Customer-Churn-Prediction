# evaluation.py
import logging
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.base import BaseEstimator
# from zenml.client import Client

# experiment_tracker = Client().active_stack.experiment_tracker

class ModelEvaluator:
    def __init__(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initializes the ModelEvaluator with model, test data, and experiment details.

        Parameters:
        
        model : BaseEstimator
            The trained model to evaluate.
        X_test : pd.DataFrame
            The test features.
        y_test : pd.Series
            The true labels for the test set.
        experiment_name : str, optional
            Name of the MLflow experiment to log metrics. Default is 'default_experiment'.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
        self.logger = logging.getLogger(__name__)

    def evaluate_model(self) -> dict:
        """
        Evaluates the model and logs metrics to MLflow.

        Returns:
        -------
        dict
            A dictionary containing accuracy, precision, recall, and f1-score.
        """
        # Set experiment explicitly
        # mlflow.set_experiment(self.experiment_name)

        # with mlflow.start_run():
        y_pred = self.model.predict(self.X_test)

        # Calculate and log evaluation metrics
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='binary'),
            "recall": recall_score(self.y_test, y_pred, average='binary'),
            "f1_score": f1_score(self.y_test, y_pred, average='binary')
        }


        # # Log evaluation metrics
        # for metric_name, metric_value in metrics.items():
        #     mlflow.log_metric(metric_name, metric_value)

        for metric_name, metric_value in metrics.items():
            # mlflow.log_metric(metric_name, metric_value)
            self.logger.info(f"{metric_name.capitalize()}: {metric_value:.4f}")

        return metrics
