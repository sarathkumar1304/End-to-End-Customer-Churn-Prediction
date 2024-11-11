import streamlit as st
import pandas as pd

# Load the metrics CSV
metrics_path = "assets/model_metrics.csv"
metrics_df = pd.read_csv(metrics_path)

def metrics_ui():
    st.title("Model Evaluation Metrics")
    st.subheader("Performance Metrics of Trained Models")

    # Display the metrics in a table
    st.dataframe(metrics_df)

    # Highlight the best model based on F1 Score
    best_model_name = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model']
    st.write(f"Best Model: {best_model_name} with an F1 Score of {metrics_df['F1 Score'].max():.2f}")
