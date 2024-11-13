import streamlit as st

def home_page():
    st.image("assets/customer.webp")
    
    st.title("Customer Churn Prediction Application 🌐 ")
    
    st.header("📝 Problem Statement")
    st.write("""
    Customer churn is a critical issue for businesses as it directly impacts profitability and growth. This application aims to predict whether a customer will leave a service or product based on historical data and behavioral patterns. 
    """)

    
    st.header(" 🎯 Objective")
    st.write("""
    The main objective of this application is to develop a machine learning model that accurately predicts customer churn. By identifying at-risk customers, businesses can take proactive measures to enhance customer retention and improve overall satisfaction.
    """)

    st.header("🛠 Technological Stack")
    st.write("""
    The application is built using the following technologies:
    - **Python**: The primary programming language used for development.
    - **Machine Learning**: Algorithms to analyze customer data and predict churn.
    - **MLOps**: Practices for deploying and maintaining machine learning models.
    - **ZenML**: A tool to create reproducible ML pipelines.
    - **MLflow**: For tracking experiments and managing model lifecycle.
    - **Streamlit**: A user-friendly UI framework for creating interactive web applications.
    - **FastAPI**: Back-end frameworks to build APIs for model interactions.
    - **Evidently Ai** : A tool for model monitering and data drift detection
    """)


    st.header(" 📝 Overview")
    st.write("""
    This design document outlines the development of a web application for predicting customer churn using a dataset that includes customer Usage Frequency, Tenure  and historical behaviors. The application will allow users to input customer data manually  and receive predictions on churn likelihood and suggested retention strategies.
    """)


    st.header(" 💪 Motivation")
    st.write("""
    Understanding and addressing customer churn can significantly enhance customer loyalty and reduce marketing costs associated with acquiring new customers. This application provides insights that help businesses to implement effective retention strategies.
    """)


    st.header(" 📈 Success Metrics")
    st.write("""
    The project's success will be measured using the following metrics:
    - Precision, Recall, and F1 Score of the churn prediction model.
    - User engagement and satisfaction with the application interface.
    - Reduction in customer churn rates observed post-implementation.
    """)

    
    st.header(" ✍ Requirements & Constraints")
    st.subheader(" ⚙️ Functional Requirements")
    st.write("""
    - Users can input customer data to receive churn predictions.
    - Users can view performance metrics of the machine learning models.
    - Users can visualize customer behavior data to derive insights.
    """)

    st.subheader(" 👨🏻‍💻 Non-Functional Requirements")
    st.write("""
    - The model should demonstrate high accuracy in predictions.
    - The application should be responsive and user-friendly.
    - User data must be handled securely.
    """)

    st.subheader("✅️ Constraints")
    st.write("""
    - The application must be built using FastAPI and Streamlit, with deployment on Docker and streamlit.
    """)

    st.header(" 🔄 Methodology")
    st.write("""
    - **Problem Statement**: Develop a model to predict customer churn based on various features.
    - **Data**: Utilize a dataset containing customer-related features such as demographics and service usage.
    - **Techniques**: Employ data preprocessing, feature engineering, model selection, training, and evaluation.
    """)

   
    st.header(" 🏛️ Architecture")
    st.write("""
    The architecture of the web application consists of:
    - A **frontend** built using Streamlit for user interaction.
    - A **backend** server implemented with FastAPI for handling requests and serving predictions.
    - A **machine learning model** for churn prediction.
    - Utilization of **Docker** for containerization.

    """)
    st.header(" 🖇️ Pipeline")
    st.image("assets/Pipeline.png", caption="Pipeline", use_column_width=True)

if __name__ == "__main__":
    home_page()
