import streamlit as st
import pandas as pd
from analysis import univariate_analysis, BivariateAnalysis, multivariate_analysis

def eda():
    st.title("Exploratory Data Analysis")

    
    data = pd.read_csv("extracted/customer_churn_dataset-training-master.csv")
    data.drop("CustomerID",axis = 1,inplace = True)
  
    data.dropna(axis=0,inplace = True,how = "all")
    st.header("Dataset Overview")
    st.dataframe(data.head())

   
    st.subheader("Select Analysis Type")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    if analysis_type == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        column = st.selectbox("Select a column for univariate analysis", data.columns, key="uni")
        plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot", "Pie Chart", "Bar Plot"], key="uni_plot")

      
        if st.button("Generate Univariate Plot", key="uni_button"):
            if column:
                univariate_analysis(data, column, plot_type)
            else:
                st.warning("Please select a column for analysis.")

    elif analysis_type == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")
        column_x = st.selectbox("Select X-axis column", data.columns, key="bi_x")
        column_y = st.selectbox("Select Y-axis column", data.columns, key="bi_y")
        plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Bar Plot", "Boxplot"], key="bi_plot")

        if st.button("Generate Bivariate Plot", key="bi_button"):
        
            if pd.api.types.is_numeric_dtype(data[column_x]) and pd.api.types.is_numeric_dtype(data[column_y]):
                analysis = BivariateAnalysis()
                analysis.numerical_vs_numerical(data, column_x, column_y, plot_type)
            elif pd.api.types.is_categorical_dtype(data[column_x]) and pd.api.types.is_categorical_dtype(data[column_y]): 
                analysis = BivariateAnalysis()
                analysis.numerical_vs_categorical(data, column_x, column_y, plot_type)
            elif pd.api.types.is_numeric_dtype(data[column_x]) and pd.api.types.is_categorical_dtype(data[column_y]):
                analysis = BivariateAnalysis()
                analysis.numerical_vs_categorical(data, column_x, column_y, plot_type)
            elif pd.api.types.is_categorical_dtype(data[column_x]) and pd.api.types.is_numeric_dtype(data[column_y]):
                analysis = BivariateAnalysis()
                analysis.numerical_vs_categorical(data, column_x, column_y, plot_type)
            elif pd.api.types.is_numeric_dtype(data[column_x]) and pd.api.types.is_numeric_dtype(data[column_y]):
                analysis = BivariateAnalysis()
                analysis.numerical_vs_numerical(data, column_x, column_y, plot_type)
            else:
                st.warning("Please select numerical columns for analysis. Only numerical data types are supported for this plot.")

    elif analysis_type == "Multivariate Analysis":
        data = pd.read_csv("/home/sarath_kumar/customer_chrun_prediction/processed_data/processed_data.csv")
        st.subheader("Multivariate Analysis")
        columns = st.multiselect("Select columns for multivariate analysis", data.columns)

        # Add an option for users to select the type of plot
        plot_type = st.selectbox(
            "Select plot type for multivariate analysis",
            ["Correlation Heatmap", "Scatter Matrix"]
        )

       
        if st.button("Generate Multivariate Plot", key="multi_button"):
            if columns:
                
                multivariate_analysis(data, columns, plot_type)
            else:
                st.warning("Please select columns for multivariate analysis.")

if __name__ == "__main__":
    eda()
