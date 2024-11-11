import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataPreprocessor with data.
        
        Parameters:
       
        data : pd.DataFrame
            The customer churn data to preprocess.
        """
        self.data = data
        self.label_encoders = {}
        logging.info("DataPreprocessor initialized with data of shape: %s", data.shape)

    def drop_customer_id(self):
        """Drop the CustomerID column if it exists in the DataFrame."""
        if 'CustomerID' in self.data.columns:
            self.data.drop(columns=['CustomerID'], inplace=True)
            logging.info("Dropped CustomerID column.")
        else:
            logging.warning("CustomerID column not found.")

    def drop_null_values(self):
        """Drop rows with any null values in the DataFrame."""
        null_count = self.data.isnull().sum().sum()
        if null_count > 0:
            self.data.dropna(inplace=True)
            logging.info("Dropped %d rows with null values.", null_count)
        else:
            logging.info("No null values to drop.")

    def encode_categorical_columns(self):
        """
        Encode categorical features: Subscription Type and Contract Length.
        Uses LabelEncoder for each specified column.
        """
        for column in ['Subscription Type', 'Contract Length']:
            if column in self.data.columns:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column].astype(str))
                self.label_encoders[column] = le
                logging.info("Encoded %s with labels: %s", column, le.classes_)
            else:
                logging.warning("%s column not found for encoding.", column)

    def map_gender(self):
        """Map Gender to binary values: Male - 1, Female - 0."""
        if 'Gender' in self.data.columns:
            self.data['Gender'] = self.data['Gender'].map({'Male': 1, 'Female': 0})
            logging.info("Mapped Gender: Male - 1, Female - 0.")
        else:
            logging.warning("Gender column not found for mapping.")

    def save_processed_data(self, output_directory='processed_data', filename='processed_data.csv'):
        """
        Save the processed data to a CSV file.
        
        Parameters:
        -----------
        output_directory : str, optional
            The directory to save the processed data (default is 'processed_data').
        filename : str, optional
            The name of the output CSV file (default is 'processed_data.csv').
        """
        os.makedirs(output_directory, exist_ok=True)
        processed_csv_path = os.path.join(output_directory, filename)
        self.data.to_csv(processed_csv_path, index=False)
        logging.info("Processed data saved to %s", processed_csv_path)

    def process_data(self) -> pd.DataFrame:
        """Execute the full preprocessing pipeline and return the processed DataFrame."""
        self.drop_customer_id()
        self.drop_null_values()
        self.encode_categorical_columns()
        self.map_gender()
        self.save_processed_data()
        logging.info("Data preprocessing completed.")
        return self.data

# Usage Example
if __name__ == '__main__':
    # df = pd.read_csv("extracted/customer_churn_dataset-training-master.csv")
    # preprocessor = DataPreprocessor(df)
    # cleaned_df = preprocessor.process_data()
    # print(cleaned_df.head())
    pass
