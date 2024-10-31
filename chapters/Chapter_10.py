import streamlit as st
import pandas as pd
import yfinance as yf
from flaml import AutoML
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models


def chapter10():
    # Initialize session state for training flag
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False

    # Title and Introduction
    st.subheader("Chapter 10: Accelerated AI and Use Cases in Investment Management")
    st.markdown("""
    ### Overview
    This app demonstrates **accelerated AI techniques** for portfolio optimization, using **AutoML** to predict returns and **Deep Reinforcement Learning** to adjust portfolio allocations.
    * It aligns with the learnings from **Chapter 10**, which emphasizes using AI to manage risk, optimize investment strategies, and enhance decision-making.
    """)

    # Sidebar for User Input (Disabled when training)
    if not st.session_state.is_training:
        st.sidebar.header("User Settings")
        assets = st.sidebar.multiselect(
            "Select Assets for Portfolio",
            ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            default=["AAPL", "GOOGL", "MSFT"],
            disabled=st.session_state.is_training
        )
        start_date = st.sidebar.date_input(
            "Start Date", pd.to_datetime("2020-01-01"), disabled=st.session_state.is_training)
        end_date = st.sidebar.date_input(
            "End Date", pd.to_datetime("2023-01-01"), disabled=st.session_state.is_training)
    else:
        st.sidebar.info("Training in progress...")

    # Fetch Financial Data for Selected Assets
    @st.cache_data
    def load_asset_data(assets, start, end):
        try:
            data = yf.download(assets, start=start, end=end)['Adj Close']
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    # Load data
    if not st.session_state.is_training:
        data = load_asset_data(assets, start_date, end_date)

    if data is not None:
        st.write("### Asset Prices")
        st.line_chart(data)

        # Calculate Returns
        returns = data.pct_change().dropna()

        # Display Returns
        st.write("### Asset Returns")
        st.line_chart(returns)

        # Data Preprocessing
        returns = data.pct_change().dropna()  # Calculate daily returns
        returns['Rolling_Mean'] = returns.mean(axis=1).rolling(
            window=5).mean()  # 5-day rolling mean
        returns['Rolling_Std'] = returns.mean(axis=1).rolling(
            window=5).std()  # 5-day rolling std
        returns['Momentum'] = returns.mean(
            axis=1).diff(5)  # Momentum over 5 days

        # Drop NaN values caused by rolling calculations
        returns = returns.dropna()

        # Prepare data for AutoML
        X = returns[['Rolling_Mean', 'Rolling_Std', 'Momentum']
                    ].values  # Use engineered features
        y = returns.mean(axis=1).values  # Aggregate returns as target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Initialize and train the AutoML model
        automl = AutoML()

        # Define AutoML settings
        automl_settings = {
            "time_budget": 60,  # Time limit in seconds
            "metric": 'mse',    # Metric to optimize (mean squared error)
            "task": 'regression',
            "log_file_name": 'automl_stock_returns.log'
        }

        # Cache AutoML Model Training
        @st.cache_resource
        def train_automl(X_train, y_train):
            automl = AutoML()
            automl_settings = {
                "time_budget": 60,  # Time limit in seconds
                "metric": 'mse',    # Metric to optimize (mean squared error)
                "task": 'regression',
                "log_file_name": 'automl_stock_returns.log'
            }
            automl.fit(X_train, y_train, **automl_settings)
            return automl

        st.session_state.is_training = True
        with st.spinner("Training AutoML Model..."):
            automl = train_automl(X_train, y_train)
        st.session_state.is_training = False

        # Make predictions
        y_pred = automl.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)

        # Plot results
        st.write("### AutoML Model Evaluation")
        st.write(f"Mean Squared Error: {mse:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual Returns', color='blue')
        plt.plot(y_pred, label='Predicted Returns', color='red')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.title('Enhanced AutoML Predicted vs. Actual Returns')
        plt.legend()
        st.pyplot(plt)

    else:
        st.info("Please select assets and date range to load data.")

    st.markdown("""
    ### How This App Relates to Chapter 10
    This app demonstrates how **accelerated AI techniques**, like AutoML and deep reinforcement learning, can be used for portfolio optimization.
    """)
