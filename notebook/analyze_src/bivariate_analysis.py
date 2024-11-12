import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import itertools
from scipy.stats import pearsonr, pointbiserialr

class BivariateAnalysis:
    def NumericalVsNumerical(self, df):
        numerical_features = df.select_dtypes(include=[float, int]).columns
        # Iterate through each pair of numerical features
        for feature1, feature2 in itertools.combinations(numerical_features, 2):
            # Calculate the Pearson correlation coefficient
            correlation, _ = pearsonr(df[feature1], df[feature2])

            # Scatter plot with trendline
            fig = px.scatter(
                df, x=feature1, y=feature2, trendline="ols",  # Ordinary Least Squares for trendline
                title=f"Scatter Plot of {feature1} vs {feature2} (Correlation: {correlation:.2f})",
                labels={feature1: feature1, feature2: feature2}
            )
            fig.update_layout(
                xaxis_title=feature1,
                yaxis_title=feature2,
                hovermode="closest"
            )
            fig.show()
    
    def NumericalVsCategorical(self, df, categorical_feature='Churn'):
        numerical_features = df.select_dtypes(include=[float, int]).columns

        # Check if the categorical feature is binary and convert it if necessary
        if df[categorical_feature].nunique() != 2:
            print(f"The categorical feature '{categorical_feature}' is not binary. Skipping correlation calculation.")
            for feature in numerical_features:
                # Box plot without correlation
                fig = px.box(
                    df, x=categorical_feature, y=feature, color=categorical_feature,
                    title=f"Box Plot of {feature} by {categorical_feature}",
                    labels={categorical_feature: categorical_feature, feature: feature}
                )
                fig.update_layout(
                    xaxis_title=categorical_feature,
                    yaxis_title=feature,
                    hovermode="x unified"
                )
                fig.show()
            return  # Exit function if not binary

        # Ensure the categorical feature is numeric (0 and 1)
        df[categorical_feature] = pd.factorize(df[categorical_feature])[0]

        # Iterate through each numerical feature
        for feature in numerical_features:
            # Remove missing values and ensure numerical data type
            valid_data = df[[feature, categorical_feature]].dropna()
            valid_data[feature] = pd.to_numeric(valid_data[feature], errors='coerce').dropna()

            # Calculate point-biserial correlation
            correlation, _ = pointbiserialr(valid_data[feature], valid_data[categorical_feature])
            title = f"Box Plot of {feature} by {categorical_feature} (Correlation: {correlation:.2f})"

            # Box plot for distribution comparison
            fig = px.box(
                valid_data, x=categorical_feature, y=feature, color=categorical_feature,
                title=title,
                labels={categorical_feature: categorical_feature, feature: feature}
            )
            fig.update_layout(
                xaxis_title=categorical_feature,
                yaxis_title=feature,
                hovermode="x unified"
            )
            fig.show()


    def NumericalVsTarget(self,df):
        numerical_features = df.select_dtypes(include=[float, int]).columns
        for feature in numerical_features:

            fig = px.box(
                df, 
                x="Churn",  
                y=feature,  
                color="Churn",  
                title=f"Distribution of {feature} by Churn Status",
                labels={"Churn": "Churn Status", feature: feature}
            )

            
            fig.update_layout(
                xaxis_title="Churn Status",
                yaxis_title=feature,
                legend_title="Churn",
                hovermode="x unified"  
            )

            
            fig.show()
    
    def CategoricalVsTarget(self,df):
        categorical_features = df.select_dtypes(include=[object]).columns
        for feature in categorical_features:
            # Create a crosstab for the feature against the target variable 'Churn'
            crosstab_data = pd.crosstab(df[feature], df['Churn'])

            crosstab_df = crosstab_data.reset_index().melt(id_vars=feature, value_name="Count")

            # Interactive bar plot with Plotly
            fig = px.bar(
                crosstab_df, 
                x=feature, 
                y="Count", 
                color="Churn",
                title=f"Churn by {feature}",
                labels={feature: feature, "Count": "Count", "Churn": "Churn Status"},
                text="Count",  
                barmode="group"  
            )


            fig.update_layout(
                xaxis_title=feature,
                yaxis_title="Count",
                legend_title="Churn",
                hovermode="x unified"  
            )

            # Show the interactive plot
            fig.show()


class FeatureImportance:
    def bivariate_analysis_with_feature_importance(self, df, target_column, model=None):
        """
        Performs bivariate analysis of each numerical feature against the target column and displays feature importance.
        
        Parameters:
        - df: DataFrame containing the data.
        - target_column: Column name of the target variable for coloring.
        - model: Trained model with `feature_importances_` attribute (e.g., RandomForest).
        """
        # Separate features and target variable
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Train a model if not provided
        if model is None:
            model = RandomForestClassifier(random_state=0)
            model.fit(X.select_dtypes(include=[np.number]), y)
        
        # Feature importance
        importance_df = pd.DataFrame({
            "Feature": X.select_dtypes(include=[np.number]).columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        # Plot feature importance
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            title="Feature Importance",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        
        fig_importance.update_layout(
            title_font=dict(size=20),
            xaxis_title="Importance Score",
            yaxis_title="Features",
            font=dict(size=12),
        )
        
        # Bivariate analysis: Plot each feature vs. target
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        fig_bivariate = px.scatter_matrix(
            df,
            dimensions=numeric_columns,
            color=target_column,
            title="Bivariate Analysis of Numerical Features",
            color_continuous_scale="Viridis",
            height=800,
        )
        
        fig_bivariate.update_traces(diagonal_visible=False)
        
        # Show plots
        fig_importance.show()
        fig_bivariate.show()

    # Example usage with a sample dataframe
    # Assuming df is your DataFrame and 'target' is your target column
    # df = pd.read_csv('your_data.csv')
    # bivariate_analysis_with_feature_importance(df, 'target_column_name')

