import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Set page config
st.set_page_config(page_title="Student Score Predictor", page_icon="📚", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("scores.csv.csv")
    return df

st.header("📚 Student Score Predictor")
st.write("This app predicts the percentage of a student based on the number of study hours using Simple Linear Regression.")

try:
    df = load_data()
    
    # Preparing the data
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Training the Algorithm
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # Sidebar for prediction
    st.sidebar.header("🎯 Make a Prediction")
    st.sidebar.write("Use this tool to predict your score!")
    hours = st.sidebar.number_input("Enter the number of hours studied:", min_value=0.0, max_value=24.0, value=9.25, step=0.25)

    if st.sidebar.button("Predict"):
        prediction = regressor.predict([[hours]])
        st.sidebar.success(f"Predicted Score for {hours} hours of study: **{prediction[0]:.2f}**")



    # Plotting the regression line
    line = regressor.coef_ * X + regressor.intercept_

    # Create columns for side-by-side graphs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualizing the Data")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='Hours', y='Scores', data=df, ax=ax)
        plt.title('Hours vs Percentage')
        plt.xlabel('Hours Studied')
        plt.ylabel('Percentage Score')
        st.pyplot(fig)

    with col2:
        st.subheader("Regression Line")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='Hours', y='Scores', data=df, ax=ax2)
        plt.plot(X, line, color='red')
        plt.title('Regression Line')
        plt.xlabel('Hours Studied')
        plt.ylabel('Percentage Score')
        st.pyplot(fig2)

    st.subheader("Model Evaluation")
    y_pred = regressor.predict(X_test)
    st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

except FileNotFoundError:
    st.error("Error: Could not find 'scores.csv.csv'. Please ensure the data file is in the same directory as the app.")
