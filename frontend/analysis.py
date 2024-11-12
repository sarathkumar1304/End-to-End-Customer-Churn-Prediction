import plotly.express as px
import streamlit as st
import pandas as pd

def univariate_analysis(data, column, plot_type):
    if plot_type == "Histogram":
        if data[column].dtype=="int64" or data[column].dtype=="float64":
            fig = px.histogram(data, x=column, title=f'Histogram of {column}')
            st.plotly_chart(fig)
        else:
            st.warning("Histograms are only suitable for numerical columns.")

    elif plot_type == "Boxplot":
        if data[column].dtype=="int64" or data[column].dtype=="float64":
            fig = px.box(data, y=column, title=f'Boxplot of {column}')
            st.plotly_chart(fig)
        else:
            st.warning("Boxplots are only suitable for numerical columns.")
    elif plot_type == "Pie Chart":
        if data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column]):
            fig = px.pie(data, names=column, title=f'Pie Chart of {column}')
            st.plotly_chart(fig)
        else:
            st.warning("Pie charts are only suitable for categorical columns.")
    elif plot_type == "Bar Plot":
        if data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column]):
            fig = px.bar(data[column].value_counts().reset_index(), x='index', y=column, title=f'Bar Plot of {column}')
            st.plotly_chart(fig)
        else:
            st.warning("Bar plots are only suitable for categorical columns.")



# def multivariate_analysis(data, columns):
#     fig = px.scatter_matrix(data, dimensions=columns, title=f'Multivariate Analysis')
#     st.plotly_chart(fig)


def multivariate_analysis(data, columns, plot_type):
    if plot_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        if len(columns) > 1:
            # Compute the correlation matrix
            correlation_matrix = data[columns].corr()

            # Create a heatmap using Seaborn and Matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please select at least two columns for a correlation heatmap.")

    elif plot_type == "Scatter Matrix":
        st.subheader("Scatter Matrix Plot")
        if len(columns) > 1:
            fig = px.scatter_matrix(data, dimensions=columns, title='Scatter Matrix Plot')
            st.plotly_chart(fig)
        else:
            st.warning("Please select at least two columns for a scatter plot matrix.")

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import itertools
from scipy.stats import pearsonr, pointbiserialr
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

class BivariateAnalysis:
    def numerical_vs_numerical(self, data, column_x, column_y, plot_type):
        plt.figure(figsize=(10, 6))
        if plot_type == "Scatter Plot":
            if data[column_x].dtype == 'int64' or data[column_x].dtype == 'float64' and data[column_y].dtype == 'int64' or data[column_y].dtype == 'float64':
                sns.scatterplot(data=data, x=column_x, y=column_y)
                plt.title(f'Scatter Plot of {column_x} vs {column_y}')
            else:
                st.warning("Scatter plots are only suitable for numerical columns.")

        elif plot_type == "Bar Plot":
            if data[column_x].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column_x]) and data[column_y].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column_y]):
                sns.barplot(data=data, x=column_x, y=column_y)
                plt.title(f'Bar Plot of {column_x} vs {column_y}')
            else:
                st.warning("Bar plots are only suitable for categorical columns.")
        elif plot_type == "Boxplot":
            if data[column_x].dtype == 'int64' or data[column_x].dtype == 'float64' and data[column_y].dtype == 'int64' or data[column_y].dtype == 'float64':
                sns.boxplot(data=data, x=column_x, y=column_y)
                plt.title(f'Boxplot of {column_x} vs {column_y}')
            else:
                st.warning("Boxplots are only suitable for numerical columns.")
        st.pyplot(plt.gcf())
        plt.clf()  


    
    def numerical_vs_categorical(df, categorical_feature='Churn'):
        numerical_features = df.select_dtypes(include=[float, int]).columns
        if df[categorical_feature].nunique() != 2:
            print(f"The categorical feature '{categorical_feature}' is not binary. Skipping correlation calculation.")
            for feature in numerical_features:
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
            return

        df[categorical_feature] = pd.factorize(df[categorical_feature])[0]
        for feature in numerical_features:
            valid_data = df[[feature, categorical_feature]].dropna()
            valid_data[feature] = pd.to_numeric(valid_data[feature], errors='coerce').dropna()
            correlation, _ = pointbiserialr(valid_data[feature], valid_data[categorical_feature])
            title = f"Box Plot of {feature} by {categorical_feature} (Correlation: {correlation:.2f})"
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

    
    def numerical_vs_target(df, target='Churn'):
        numerical_features = df.select_dtypes(include=[float, int]).columns
        for feature in numerical_features:
            fig = px.box(
                df, 
                x=target,  
                y=feature,  
                color=target,  
                title=f"Distribution of {feature} by {target} Status",
                labels={target: f"{target} Status", feature: feature}
            )
            fig.update_layout(
                xaxis_title=f"{target} Status",
                yaxis_title=feature,
                legend_title=target,
                hovermode="x unified"
            )
            fig.show()

    
    def categorical_vs_target(df, target='Churn'):
        categorical_features = df.select_dtypes(include=[object]).columns
        for feature in categorical_features:
            crosstab_data = pd.crosstab(df[feature], df[target])
            crosstab_df = crosstab_data.reset_index().melt(id_vars=feature, value_name="Count")
            fig = px.bar(
                crosstab_df, 
                x=feature, 
                y="Count", 
                color=target,
                title=f"{target} by {feature}",
                labels={feature: feature, "Count": "Count", target: f"{target} Status"},
                text="Count",
                barmode="group"
            )
            fig.update_layout(
                xaxis_title=feature,
                yaxis_title="Count",
                legend_title=target,
                hovermode="x unified"
            )
            fig.show()

    def feature_importance(df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = RandomForestClassifier(random_state=0)
        model.fit(X.select_dtypes(include=[np.number]), y)

        importance_df = pd.DataFrame({
            "Feature": X.select_dtypes(include=[np.number]).columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

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
        fig_importance.show()
