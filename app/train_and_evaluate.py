import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from xgboost import XGBClassifier
import logging

# Configure logging
logging.basicConfig(
    filename='/home/sarath_kumar/customer_chrun_prediction/training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting training script...")

try:
    # Load dataset
    data = pd.read_csv("/home/sarath_kumar/customer_chrun_prediction/processed_data/processed_data.csv")
    logging.info("Dataset loaded successfully.")

    # Split data into features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    logging.info("Data split into features and target.")

    # Dictionary to hold model instances and names
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500,solver='saga'),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(),
    }

    # List to store evaluation metrics
    metrics_list = []

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logging.info("Data split into training and testing sets.")

    # Train and evaluate models
    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        logging.info(f"{model_name} training completed.")

        y_pred = model.predict(X_test)
        logging.info(f"{model_name} prediction completed.")

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        logging.info(f"{model_name} evaluation metrics calculated.")

        # Append metrics to the list
        metrics_list.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # Create a DataFrame from the metrics
    metrics_df = pd.DataFrame(metrics_list)

    # Save metrics to a CSV file
    metrics_df.to_csv("/home/sarath_kumar/customer_chrun_prediction/model_metrics.csv", index=False)
    logging.info("Metrics saved to CSV successfully.")

    # Optionally, save trained models for further analysis or use
    # for model_name, model in models.items():
    #     joblib.dump(model, f"/home/sarath_kumar/customer_chrun_prediction/models/{model_name}.pkl")
    #     logging.info(f"{model_name} saved to file.")

    logging.info("Training script completed successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    raise
