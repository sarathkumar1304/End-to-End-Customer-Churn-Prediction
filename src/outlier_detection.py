import pandas as pd
import numpy as np
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetector:

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the OutlierDetector with data.
        
        Parameters:
        
        data : pd.DataFrame
            The data for outlier detection.
        """
        self.data = data
        logging.info("OutlierDetector initialized with data of shape: %s", data.shape)

    def z_score_outlier_detection(self, threshold: float = 3.0) -> pd.DataFrame:
        """Detect outliers using Z-Score method."""
        logging.info("Calculating Z-Scores for outlier detection.")
        z_scores = np.abs((self.data - self.data.mean()) / self.data.std())
        outliers = (z_scores > threshold)
        logging.info("Detected %d outliers using Z-Score method.", outliers.sum().sum())
        return self.data[~outliers.any(axis=1)]  # Return DataFrame without outliers

    def iqr_outlier_detection(self) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        logging.info("Calculating IQR for outlier detection.")
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))
        logging.info("Detected %d outliers using IQR method.", outlier_condition.sum().sum())
        return self.data[~outlier_condition.any(axis=1)]  # Return DataFrame without outliers

    def run_outlier_detection(self) -> pd.DataFrame:
        """Run all outlier detection methods and return cleaned data."""
        logging.info("Starting outlier detection steps.")
        
        # Select only numerical columns for outlier detection
        numerical_data = self.data.select_dtypes(include=[np.number])  # Include all numerical columns
        logging.info("Selected numerical columns for outlier detection: %s", numerical_data.columns.tolist())
        
        # Z-Score Method
        cleaned_data_z = self.z_score_outlier_detection()
        
        # IQR Method
        cleaned_data_iqr = self.iqr_outlier_detection()

        logging.info("Outlier detection completed.")
        
        # Return a dictionary of cleaned data
        return  cleaned_data_iqr

# Usage Example
if __name__ == '__main__':
    # Sample data
    # try:
    #     df = pd.read_csv("extracted/customer_churn_dataset-training-master.csv")
    #     logging.info("Loaded dataset with shape: %s", df.shape)

    #     # Initialize the outlier detector
    #     detector = OutlierDetector(df)

    #     # Run the outlier detection
    #     cleaned_data = detector.run_outlier_detection()

    #     # Display the cleaned DataFrames
    #     logging.info("Cleaned Data (Z-Score):")
    #     print(cleaned_data["z_score_cleaned"].head())
        
    #     logging.info("Cleaned Data (IQR):")
    #     print(cleaned_data["iqr_cleaned"].head())
        
    # except Exception as e:
    #     logging.error("An error occurred: %s", e)
    pass
