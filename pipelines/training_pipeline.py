from zenml import pipeline 
from zenml import Model
from steps.data_ingestion_step import data_ingestion_step
from steps.data_preprocessing_step import data_preprocessing_step
from steps.outlier_detection_step import outlier_detection_step 
from steps.data_splitting_step import data_splitter_step
from steps.model_building_step import model_builder_step
from steps.model_evaluation_step import model_evaluation_step
import logging
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)
@pipeline(
    model=Model(
        name="customer_churn_prediction",
    )
)
def training_pipeline():
    """Defines an end-to-end machine learning pipeline for customer churn prediction."""
    
    """Defines an end-to-end machine learning pipeline."""
    # Data Ingestion Step
    # Load raw data from the specified file path
    raw_data = data_ingestion_step("/home/sarath_kumar/customer_chrun_prediction/data/customer_churn_dataset-training-master.csv.zip")
    
    # Data Preprocessing Step
    # Preprocess the raw data to clean and format it appropriately
    cleaned_data = data_preprocessing_step(raw_data)
    
    # Outlier Detection Step
    outlier_removed_data = outlier_detection_step(cleaned_data)
    
    # Data Splitting Step
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = data_splitter_step(outlier_removed_data, target_column="Churn")
    
    # Model Building Step
    # Build and train the model using the training data
    model = model_builder_step(model_name="xgboost", X_train=X_train, y_train=y_train)

    metrics = model_evaluation_step(model, X_test, y_test)
    
    # Return the trained model
    return model


if __name__ == "__main__":
    training_pipeline()  
