import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math

class UnivariateAnalysis:
    def OutlierDetection(self, df):
        numerical_features = df.select_dtypes(include=[float, int]).columns
        for feature in numerical_features:
            fig = px.box(df, y=feature, title=f"Boxplot of {feature}")
            fig.show()

    def NumericalDistribution(self, df, bins=20):
        numerical_features = df.select_dtypes(include=[float, int]).columns
        for feature in numerical_features:
            fig = px.histogram(df, x=feature, nbins=bins, title=f"Histogram of {feature}", marginal="box")
            fig.update_traces(marker=dict(color="skyblue", line=dict(color="black", width=1)))
            fig.show()

    def CategoricalUnivariateAnalysis(self, df):
        categorical_features = df.select_dtypes(include=[object]).columns
        for feature in categorical_features:
            # Create a DataFrame from value_counts with explicit column names
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, "Count"]  # Rename columns for clarity

            fig = px.bar(
                value_counts, 
                x=feature, 
                y="Count", 
                title=f"Bar Plot of {feature}", 
                labels={feature: feature, "Count": "Count"}
            )
            fig.show()

    def PieChart(self, df):
        categorical_features = df.select_dtypes(include=[object]).columns
        for feature in categorical_features:
            fig = px.pie(df, names=feature, title=f"Pie Chart of {feature}", hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.show()
