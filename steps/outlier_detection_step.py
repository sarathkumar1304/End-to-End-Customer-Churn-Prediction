import pandas as pd
from src.outlier_detection import OutlierDetector  
from zenml import step


@step
def outlier_detection_step(data: pd.DataFrame) -> pd.DataFrame:
    
    """
    A ZenML step to detect outliers in a given DataFrame using the OutlierDetector class.

    Parameters
    data : pd.DataFrame
        The DataFrame containing the data to detect outliers in.

    Returns
    
    pd.DataFrame
        The DataFrame with outliers removed.
    """

    detector = OutlierDetector(data)
    cleaned_data = detector.run_outlier_detection()
    return cleaned_data
