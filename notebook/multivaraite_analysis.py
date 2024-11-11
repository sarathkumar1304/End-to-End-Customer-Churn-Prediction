import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import pandas as pd

class MultivariateAnalysis:
    def __init__(self, df:pd.DataFrame, target_column:str):
        self.df = df
        self.target_column = target_column

    def correlation_heatmap(self):
        """Generates an interactive heatmap of correlations between numerical features with enhanced features."""
        
        # Calculate correlation matrix
        correlation_matrix = self.df.corr()
        
        # Create an interactive heatmap
        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",  # Shows values with two decimal points
            color_continuous_scale="Viridis",
            title="Correlation Heatmap of Numerical Features",
            aspect="auto"
        )
        
        # Update layout for better interactivity
        fig.update_layout(
            autosize=False,
            width=900,  # Adjust width
            height=800,  # Adjust height
            margin=dict(l=50, r=50, t=50, b=50),  # Set margins
            hovermode="closest",  # Show hover information for closest point
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=np.linspace(-1, 1, 5),  # Tick values from -1 to 1
                lenmode="fraction",
                len=0.75  # Colorbar length
            )
        )
        
        # Add interactivity: hovering shows feature names and correlation value
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                fig.add_annotation(
                    go.layout.Annotation(
                        x=j,
                        y=i,
                        showarrow=False,
                        text=f"{correlation_matrix.iloc[i, j]:.2f}",
                        font=dict(color="black", size=10),
                        opacity=0.8
                    )
                )
        
        fig.show()

    def pairplot(self, marker_size=5, orientation='v', diagonal_visible=False):
        """
        Generates an interactive pairplot to visualize relationships between pairs of numerical variables.
        
        Parameters:
        - df: DataFrame containing the data.
        - target_column: Column name of the target variable for coloring.
        - marker_size: Size of the markers in the plot (default: 5).
        - orientation: Orientation of labels ('h' for horizontal, 'v' for vertical, default: 'v').
        - diagonal_visible: Boolean to show/hide diagonal plots (default: False).
        """
        
        # Select only numerical columns for the scatter matrix
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        fig = px.scatter_matrix(
            self.df,
            dimensions=numeric_columns,
            color=self.target_column,
            title="Enhanced Pairplot of Numerical Features",
            color_continuous_scale="Viridis",
            height=800,
            hover_data=self.df.columns,  # Add all columns for hover information
        )
        
        fig.update_traces(
            diagonal_visible=diagonal_visible,
            marker=dict(size=marker_size)
        )
        
        # Update axis labels orientation
        fig.update_layout(
            title_font=dict(size=20),
            font=dict(size=12),
            xaxis_title=None,
            yaxis_title=None,
        )
        
        # Adjust axis tick labels orientation for each individual axis
        for axis_name in fig.layout:
            if "xaxis" in axis_name or "yaxis" in axis_name:
                fig.layout[axis_name].tickangle = 0 if orientation == 'h' else -90
        
        fig.show()

    
    def interactive_3d_scatter_plot(self):
        """Generates a dynamic interactive 3D scatter plot for any three selected numerical features."""
        numerical_features = self.df.select_dtypes(include=[np.number]).columns

        # Create dropdowns for feature selection
        feature1_dropdown = widgets.Dropdown(options=numerical_features, description="X-axis:")
        feature2_dropdown = widgets.Dropdown(options=numerical_features, description="Y-axis:")
        feature3_dropdown = widgets.Dropdown(options=numerical_features, description="Z-axis:")

        # Update plot based on selected features
        def update_3d_plot(feature1:str, feature2:str, feature3:str):
            if feature1 != feature2 and feature1 != feature3 and feature2 != feature3:
                fig = px.scatter_3d(
                    self.df, x=feature1, y=feature2, z=feature3,
                    color=self.target_column, color_continuous_scale="Viridis",
                    title=f"3D Scatter Plot of {feature1}, {feature2}, and {feature3}"
                )
                fig.update_traces(marker=dict(size=5))
                fig.show()

        
        display(widgets.interactive(update_3d_plot, feature1=feature1_dropdown, feature2=feature2_dropdown, feature3=feature3_dropdown))


    def scatter_plot_with_hue(self):
        """Generates interactive scatter plots with hue as target variable for each pair of numerical variables."""
        numerical_features = self.df.select_dtypes(include=[np.number]).columns
        feature1_dropdown = widgets.Dropdown(
            options=numerical_features, description="Feature 1:"
        )
        feature2_dropdown = widgets.Dropdown(
            options=numerical_features, description="Feature 2:"
        )

        def update_scatter(feature1, feature2):
            if feature1 != feature2:
                fig = px.scatter(
                    self.df, x=feature1, y=feature2, color=self.target_column,
                    title=f"{feature1} vs {feature2} with {self.target_column} as hue",
                    color_continuous_scale="Viridis"
                )
                fig.show()

        # Link the update function to the dropdowns
        widgets.interactive(update_scatter, feature1=feature1_dropdown, feature2=feature2_dropdown)

        # Display the dropdowns in a vertical box
        display(widgets.VBox([feature1_dropdown, feature2_dropdown]))

    def three_dimensional_scatter_plot(self, feature1:str, feature2:str, feature3:str):
        """Generates an interactive 3D scatter plot for three numerical features."""
        fig = px.scatter_3d(
            self.df, x=feature1, y=feature2, z=feature3,
            color=self.target_column, color_continuous_scale="Viridis",
            title=f"3D Scatter Plot of {feature1}, {feature2}, and {feature3}"
        )
        fig.update_traces(marker=dict(size=5))
        fig.show()

# Example Usage:
# Assuming `df` is your DataFrame and "Churn" is the target column
# analysis = MultivariateAnalysis(df, "Churn")
# analysis.correlation_heatmap()
# analysis.pairplot()
# analysis.scatter_plot_with_hue()
# analysis.three_dimensional_scatter_plot("Feature1", "Feature2", "Feature3")
