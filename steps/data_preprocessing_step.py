import pandas as pd
from src.data_preprocessing import DataPreprocessor 
from zenml import step

@step
def data_preprocessing_step(data: pd.DataFrame) -> pd.DataFrame:
    
    """
    A ZenML step to preprocess the data using the DataPreprocessor class.

    Parameters:
    ----------
    data : pd.DataFrame
        The DataFrame containing the customer churn data to preprocess.

    Returns:
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    preprocessor = DataPreprocessor(data)
    processed_data = preprocessor.process_data()
    return processed_data
