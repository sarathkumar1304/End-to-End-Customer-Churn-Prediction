import zipfile
import os
import logging


# Configure logging


class ZipExtractor:

    def __init__(self, zip_path, output_folder="extracted"):
        """
        Initializes the ZipExtractor with the zip file path and output folder.
        
        Parameters:
        zip_path : str
            Path to the zip file.
        output_folder : str, optional
            Folder where extracted files will be saved (default is "extracted").
        """
        self.zip_path = zip_path
        self.output_folder = output_folder
        logging.info(f"Initialized ZipExtractor with zip_path: {zip_path} and output_folder: {output_folder}")

    def extract_files(self):
        """
        Extracts files from the zip archive. If the zip file contains CSV files,
        they are extracted to the specified output folder. Handles errors if
        the zip file is corrupted or the path is invalid.

        Returns:
        None
        """
        logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging.log",'w+'),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)
        try:
            # Check if output folder exists; create if it doesn't
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                logging.info(f"Created output folder: {self.output_folder}")
            
            # Open the zip file and start extraction
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                csv_files = [file for file in zip_ref.namelist() if file.endswith('.csv')]
                
                if not csv_files:
                    logging.warning("No CSV files found in the zip archive.")
                    return

                for file in csv_files:
                    # Extract only CSV files
                    zip_ref.extract(file, self.output_folder)
                    logging.info(f"Extracted {file} to {self.output_folder}")

        except zipfile.BadZipFile:
            logging.error("Error: The zip file is corrupted or invalid.")
        except FileNotFoundError:
            logging.error("Error: The zip file was not found at the specified path.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

# Usage Example:
if __name__ == "__main__":
    # Create an instance of ZipExtractor
    # extractor = ZipExtractor(zip_path='data/customer_churn_dataset-training-master.csv.zip')
    # # Call the extract_files method
    # extractor.extract_files()
    pass
