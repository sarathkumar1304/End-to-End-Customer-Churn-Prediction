import pandas as pd
from zenml import step
from src.data_splitting import DataSplitter
from typing_extensions import Tuple
@step
def data_splitter_step(df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    ZenML step to split the dataframe using the DataSplitter class.

    Parameters:

    df : pd.DataFrame
        The input dataframe to be split.
    target_column : str
        The name of the target column in the dataframe.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.

    Returns:

    Tuple of X_train, X_test, y_train, y_test
    """
    # Initialize the DataSplitter
    splitter = DataSplitter(df, target_column, test_size)
    
    # Perform the train-test split
    X_train, X_test, y_train, y_test = splitter.split_data()
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == '__main__':
    # # Sample data for testing
    # df = pd.read_csv('data.csv')
    # target_column = 'target'
    
    # # Call the ZenML step
    # X_train, X_test, y_train, y_test = data_splitter_step(df, target_column)
    
    # # Display the results
    # print("X_train:\n", X_train)
    # print("X_test:\n", X_test)
    # print("y_train:\n", y_train)
    # print("y_test:\n", y_test)
    pass