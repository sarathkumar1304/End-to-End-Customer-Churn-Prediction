from zenml import step
import pandas as pd
import logging
from src.data_ingestion import DataIngestion




@step
def data_ingestion_step(file_path:str)->pd.DataFrame:

    """
    Data ingestion step.

    This step takes in a pandas DataFrame and performs the necessary data ingestion steps.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The ingested data.
    """
    logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging.log",'w+'),
        logging.StreamHandler()  
    ]
)
    data_ingest = DataIngestion()
    data = data_ingest.data_ingestion(file_path)
    return data


