import os
import pandas as pd
from .zip_extractor import ZipExtractor  
import logging




class DataIngestion:
    def data_ingestion(self,zip_path: str) -> pd.DataFrame:
        """
        Function to extract a zip file, read the CSV data into a DataFrame, 
        and handle any extraction or reading errors gracefully.
        
        Parameters:
        
        zip_path : str
            Path to the zip file containing the CSV files.
            
        Returns:
        
        pd.DataFrame
            A pandas DataFrame containing the data from the CSV file.
        
        Raises:
        
        Exception if extraction or CSV reading fails.
        """
        
        logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logging.log",'w+'), 
            logging.StreamHandler() 
    ]
)
        
        try:
            # Initialize ZipExtractor with specified zip path
            extractor = ZipExtractor(zip_path=zip_path)
            logging.info(f"Initialized ZipExtractor with path: {zip_path}")
            
            # Extract files and ensure CSVs are in the specified folder
            extractor.extract_files()
            logging.info("CSV Files are extracted from {zip_path}.")
            
            # Get the output folder where CSV files are extracted
            output_folder = extractor.output_folder
            logging.info(f"Extracted files are located in: {output_folder}")
            
            # Find extracted CSV files in the output folder
            csv_files = [file for file in os.listdir(output_folder) if file.endswith('.csv')]
            
            if not csv_files:
                logging.error("No CSV files found in the extracted folder.")
                raise FileNotFoundError("No CSV files found in the extracted folder.")
            
            # Read the first CSV file found into a DataFrame
            csv_path = os.path.join(output_folder, csv_files[0])
            data = pd.read_csv(csv_path)
            logging.info(f"Successfully loaded data from {csv_files[0]}")
            # logging.debug(f"Data preview:\n{data.head()}")
            return data

        except FileNotFoundError as e:
            logging.error(f"Error: {e}")
            raise
        except pd.errors.EmptyDataError:
            logging.error("Error: The CSV file is empty.")
            raise
        except pd.errors.ParserError:
            logging.error("Error: The CSV file contains parsing errors.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during data ingestion: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    # data_ingest = DataIngestion()
    # try:
    #     df = data_ingest.data_ingestion("data/raw/customer_churn_dataset-training-master.csv.zip")
    # except Exception as e:
    #     logging.error(f"Data ingestion failed: {e}")
    pass
