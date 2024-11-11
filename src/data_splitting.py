import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataSplitter:
    def __init__(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the DataSplitter with a DataFrame and parameters for splitting.

        Parameters:
        df : pd.DataFrame
            The input dataframe to be split.
        target_column : str
            The name of the target column in the dataframe.
        test_size : float, optional
            The proportion of the dataset to include in the test split. Default is 0.2.
        random_state : int, optional
            Controls the shuffling applied to the data before splitting. Default is 42.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Check if target_column exists in the DataFrame
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' does not exist in the DataFrame.")
        
        logging.info("DataSplitter initialized successfully.")

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataframe into train and test sets.

        Returns:
        Tuple of X_train, X_test, y_train, y_test:
        X_train : pd.DataFrame
            Training set features.
        X_test : pd.DataFrame
            Testing set features.
        y_train : pd.Series
            Training set target variable.
        y_test : pd.Series
            Testing set target variable.
        """
        logging.info(f"Starting train-test split with test_size={self.test_size} and random_state={self.random_state}.")

        
        X = self.df.drop(columns=[self.target_column], axis=1)
        y = self.df[self.target_column]

       
        logging.info(f"Feature set shape: {X.shape}")
        logging.info(f"Target set shape: {y.shape}")

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        
        logging.info(f"Train feature set shape: {X_train.shape}")
        logging.info(f"Test feature set shape: {X_test.shape}")
        logging.info(f"Train target set shape: {y_train.shape}")
        logging.info(f"Test target set shape: {y_test.shape}")
        logging.info("Train-test split completed successfully.")

        return X_train, X_test, y_train, y_test
