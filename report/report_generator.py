import pandas as pd
import numpy as np 
# from backend.config.config import REPORTS_DIR, ARTIFACTS_DIR, DATA_DIR
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset,DataQualityPreset,ClassificationPreset
from evidently import ColumnMapping
from sklearn.model_selection import train_test_split
import shutil
from evidently.metrics import ClassificationQualityMetric,ClassificationQualityByClass,ClassificationConfusionMatrix




def data_report():
    
    ref_data = pd.read_csv("test_data/customer_churn_dataset-testing-master.csv")
    current_data = pd.read_csv("test_data/customer_churn_dataset-training-master.csv")

    ref_data.drop('CustomerID',axis=1,inplace=True)
    current_data.drop('CustomerID',axis=1,inplace=True)

    ref_data.dropna(inplace=True,axis=0,how='any')
    current_data.dropna(inplace=True,axis=0,how='any')
  
    ref_data['Gender'] = ref_data['Gender'].map({'Female': 0, 'Male': 1})
    current_data['Gender'] = current_data['Gender'].map({'Female': 0, 'Male': 1})

    label_encoder = LabelEncoder()
    ref_data['Subscription Type'] = label_encoder.fit_transform(ref_data['Subscription Type'])
    current_data['Subscription Type'] = label_encoder.transform(current_data['Subscription Type'])

    ref_data['Contract Length'] = label_encoder.fit_transform(ref_data['Contract Length'])
    current_data['Contract Length'] = label_encoder.transform(current_data['Contract Length'])
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    
    report.run(reference_data=ref_data, current_data=current_data,column_mapping=None)

    frontend_report_dir = Path("frontend/reports")
    frontend_report_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(str(frontend_report_dir / "report.html"))



def model_report_1():
    ref_data = pd.read_csv("test_data/customer_churn_dataset-testing-master.csv")
    current_data = pd.read_csv("test_data/customer_churn_dataset-training-master.csv")

    
    ref_data.drop("CustomerID", axis=1, inplace=True)
    current_data.drop("CustomerID", axis=1, inplace=True)

    ref_data.dropna(inplace=True, axis=0, how='any')
    current_data.dropna(inplace=True, axis=0, how='any')
    numerical_features = list(ref_data.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = list(ref_data.select_dtypes(include=[object]).columns)



    ref_data['Gender'] = ref_data['Gender'].map({'Female': 0, 'Male': 1})
    current_data['Gender'] = current_data['Gender'].map({'Female': 0, 'Male': 1})

    label_encoder = LabelEncoder()
    ref_data['Subscription Type'] = label_encoder.fit_transform(ref_data['Subscription Type'])
    current_data['Subscription Type'] = label_encoder.transform(current_data['Subscription Type'])

    ref_data['Contract Length'] = label_encoder.fit_transform(ref_data['Contract Length'])
    current_data['Contract Length'] = label_encoder.transform(current_data['Contract Length'])


   
    with open('backend/artifacts/Decision Tree.pkl', 'rb') as f:
        model = joblib.load(f)

    
    ref_x = ref_data.drop("Churn", axis=1)
    ref_y = ref_data["Churn"]
    current_x = current_data.drop("Churn", axis=1)
    current_y = current_data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(ref_x, ref_y, test_size=0.2, random_state=0)


   
    ref_pred_prob = model.predict_proba(X_test)
    cur_pred_prob = model.predict_proba(current_x)

    
    ref_pred = pd.DataFrame(ref_pred_prob[:, 1], columns=['Prediction'])
    cur_pred = pd.DataFrame(cur_pred_prob[:, 1], columns=['Prediction'])

    
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    ref_merged = pd.concat([X_test, y_test, ref_pred], axis=1)

    current_x.reset_index(drop=True, inplace=True)
    current_y.reset_index(drop=True, inplace=True)
    cur_merged = pd.concat([current_x, current_y, cur_pred], axis=1)

    
    cm = ColumnMapping()
    cm.target = "Churn"
    cm.prediction = "Prediction"
    cm.numerical_features = numerical_features
    cm.categorical_features = categorical_features

    
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=ref_merged, current_data=cur_merged, column_mapping=cm)
    # report.show()

    
    frontend_report_dir = Path("frontend/reports")
    frontend_report_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(str(frontend_report_dir/"model_report_1.html"))



def model_report_2():


    ref_data = pd.read_csv("test_data/customer_churn_dataset-testing-master.csv")
    current_data = pd.read_csv("test_data/customer_churn_dataset-training-master.csv")

    
    ref_data.drop("CustomerID", axis=1, inplace=True)
    current_data.drop("CustomerID", axis=1, inplace=True)
    ref_data.dropna(inplace=True)
    current_data.dropna(inplace=True)

    numerical_features = list(ref_data.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = list(ref_data.select_dtypes(include=[object]).columns)

   
    ref_data['Gender'] = ref_data['Gender'].map({'Female': 0, 'Male': 1})
    current_data['Gender'] = current_data['Gender'].map({'Female': 0, 'Male': 1})

    label_encoder = LabelEncoder()
    ref_data['Subscription Type'] = label_encoder.fit_transform(ref_data['Subscription Type'])
    current_data['Subscription Type'] = label_encoder.transform(current_data['Subscription Type'])
    ref_data['Contract Length'] = label_encoder.fit_transform(ref_data['Contract Length'])
    current_data['Contract Length'] = label_encoder.transform(current_data['Contract Length'])

   
    with open('backend/artifacts/Random Forest.pkl', 'rb') as f:
        model = joblib.load(f)

    
    ref_x = ref_data.drop("Churn", axis=1)
    ref_y = ref_data["Churn"]
    current_x = current_data.drop("Churn", axis=1)
    current_y = current_data["Churn"]

    
    X_train, X_test, y_train, y_test = train_test_split(ref_x, ref_y, test_size=0.2, random_state=42)

  
    ref_pred = model.predict(X_test)
    cur_pred = model.predict(current_x)

    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    ref_merged = pd.concat([X_test, y_test, pd.Series(ref_pred, name="Prediction")], axis=1)

    current_x.reset_index(drop=True, inplace=True)
    current_y.reset_index(drop=True, inplace=True)
    cur_merged = pd.concat([current_x, current_y, pd.Series(cur_pred, name="Prediction")], axis=1)

    # Column mapping for Evidently
    cm = ColumnMapping()
    cm.target = "Churn"
    cm.prediction = "Prediction"
    cm.numerical_features = numerical_features
    cm.categorical_features = categorical_features

    # Run Evidently report with ClassificationQualityMetric
    report = Report(metrics=[ClassificationQualityMetric(),ClassificationQualityByClass(),ClassificationConfusionMatrix()])
    report.run(reference_data=ref_merged, current_data=cur_merged, column_mapping=cm)
    report.show()

    # Save report to frontend directory
    frontend_report_dir = Path("frontend/reports")
    frontend_report_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(str(frontend_report_dir/"model_report_2.html"))

def generate_report():
    data_report()
    model_report_1()
    model_report_2()


    src = Path("frontend/reports")
    dst = "backend/reports"

    if Path(dst).exists():
        shutil.rmtree(dst)

    
    shutil.copytree(src, dst)

generate_report()
