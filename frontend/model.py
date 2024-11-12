import streamlit as st
import pandas as pd
import streamlit.components.v1 as components


metrics_path = "assets/model_metrics.csv"
metrics_df = pd.read_csv(metrics_path)

def metrics_ui():
    st.title("Model Evaluation Metrics")
    st.subheader("Performance Metrics of Trained Models")

    
    st.dataframe(metrics_df)

    best_model_name = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model']
    st.write(f"Best Model: {best_model_name} with an F1 Score of {metrics_df['F1 Score'].max():.2f}")

    st.subheader("Dataset Drift")
    html = "frontend/reports/report.html"
    with open(html,'r') as f:
        html_data= f.read()
    
    st.components.v1.html(html_data,scrolling = True,height=700,width= 800)


    st.subheader("Decison Tree Model Report")
    html = "frontend/reports/model_report_1.html"
    with open(html,'r') as f:
        html_data= f.read()
    
    st.components.v1.html(html_data,scrolling = True,height=700,width= 800)

    st.subheader("RandomForest Model  Drift")
    html = "frontend/reports/model_report_2.html"
    with open(html,'r') as f:
        html_data= f.read()
    
    st.components.v1.html(html_data,scrolling = True,height=700,width= 800)


    
