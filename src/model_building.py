import logging
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelBuilding:
    def logistic_regression(self, X_train, y_train) -> Any:
        """Initialize, fit, and return a Logistic Regression model."""
        logger.info("Initializing Logistic Regression model...")
        model = LogisticRegression()
        model.fit(X_train, y_train)
        logger.info("Logistic Regression model trained successfully.")
        return model

    def xgboost(self, X_train, y_train) -> Any:
        """Initialize, fit, and return a Naive Bayes classifier model."""
        logger.info("Initializing xgboost model...")
        model = XGBClassifier()
        model.fit(X_train, y_train)
        logger.info("xgboost model trained successfully.")
        return model

    def random_forest(self, X_train, y_train) -> Any:
        """Initialize, fit, and return a Random Forest classifier model."""
        logger.info("Initializing Random Forest model...")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully.")
        return model

    def decision_tree(self, X_train, y_train) -> Any:
        """Initialize, fit, and return a Decision Tree classifier model."""
        logger.info("Initializing Decision Tree model...")
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        logger.info("Decision Tree model trained successfully.")
        return model

    def get_model(self, model_name: str, X_train, y_train) -> Any:
        

        """
        Initialize, fit, and return a machine learning model by name.

        Parameters:
        
        model_name : str
            The name of the model to create.
        X_train : pd.DataFrame
            The feature data to train the model on.
        y_train : pd.Series
            The target data to train the model on.

        Returns :
        model : Any
            The trained model instance.

        Raises:
        ValueError
            If the model name is not recognized.
        """
        if model_name == "logistic_regression":
            return self.logistic_regression(X_train, y_train)
        elif model_name == "xgboost":
            return self.xgboost(X_train, y_train)
        elif model_name == "random_forest":
            return self.random_forest(X_train, y_train)
        elif model_name == "decision_tree":
            return self.decision_tree(X_train, y_train)
        else:
            logger.error(f"Model '{model_name}' not recognized.")
            raise ValueError(f"Model '{model_name}' not recognized.")
