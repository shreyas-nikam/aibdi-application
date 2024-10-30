import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set Streamlit page configuration


def chapter7():
    # App Title
    st.title('Advanced Trading Strategy Simulation')
    st.divider()
    st.markdown("""
    This app allows users to simulate trading strategies with **dynamic predictions of transaction costs**. Users can adjust parameters like **trade volume**, **market impact**, and **slippage** to explore different execution outcomes.
    This app aligns with the concepts from **Chapter 7: Machine Learning and Big Data Trade Execution Support**, where the focus is on using ML to optimize transaction costs and improve execution strategies.
    """)

    # Sidebar - User Inputs
    st.sidebar.header('Adjust Simulation Parameters')

    # User Inputs for Trade Simulation
    trade_volume = st.sidebar.number_input(
        'Total Trade Volume', min_value=1000, max_value=1000000, value=100000, step=1000)
    strategy_type = st.sidebar.selectbox('Execution Strategy Type', [
                                         'TWAP', 'VWAP', 'POV (Proportion of Volume)'], default='VWAP')
    market_impact = st.sidebar.slider(
        'Market Impact Factor', min_value=0.0, max_value=0.1, value=0.05, step=0.01)
    slippage = st.sidebar.slider(
        'Slippage Factor', min_value=0.0, max_value=0.05, value=0.01, step=0.01)

    # Simulate Market Conditions
    st.sidebar.subheader('Simulated Market Conditions')
    market_condition = st.sidebar.selectbox(
        'Market Condition', ['Stable', 'Rising', 'Falling'])
    execution_period = st.sidebar.slider(
        'Execution Period (Slices)', min_value=5, max_value=50, value=10)

    # Generate Simulated Market Data
    np.random.seed(42)
    if market_condition == 'Stable':
        market_volumes = np.random.uniform(5000, 7000, execution_period)
        market_prices = np.linspace(100, 100, execution_period)
    elif market_condition == 'Rising':
        market_volumes = np.linspace(5000, 10000, execution_period)
        market_prices = np.linspace(100, 110, execution_period)
    else:
        market_volumes = np.linspace(10000, 5000, execution_period)
        market_prices = np.linspace(110, 100, execution_period)

    # Define Execution Strategies
    def twap_execution(volume, period):
        return np.full(period, volume / period)

    def vwap_execution(volume, market_volumes):
        weights = market_volumes / market_volumes.sum()
        return volume * weights

    def pov_execution(volume, market_volumes, pov_ratio=0.1):
        return market_volumes * pov_ratio

    # Calculate Execution Volume based on Selected Strategy
    if strategy_type == 'TWAP':
        execution_volumes = twap_execution(trade_volume, execution_period)
    elif strategy_type == 'VWAP':
        execution_volumes = vwap_execution(trade_volume, market_volumes)
    else:
        execution_volumes = pov_execution(trade_volume, market_volumes)

    # Calculate Transaction Costs
    transaction_costs = (market_impact * execution_volumes) + \
        (slippage * market_prices)

    # Data for Visualization
    df = pd.DataFrame({
        'Time Slice': range(1, execution_period + 1),
        'Market Volume': market_volumes,
        'Market Price': market_prices,
        'Execution Volume': execution_volumes,
        'Transaction Cost': transaction_costs
    })

    # Model Transaction Cost Predictions (Basic Linear Regression)
    X = df[['Market Volume', 'Market Price']]
    y = df['Transaction Cost']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted_costs = model.predict(X)

    # Display Simulation Results
    st.subheader('Simulation Results')
    st.write(f"**Total Trade Volume**: {trade_volume} units")
    st.write(f"**Execution Strategy**: {strategy_type}")
    st.write(f"**Market Condition**: {market_condition}")
    st.write(f"**Market Impact Factor**: {market_impact}")
    st.write(f"**Slippage Factor**: {slippage}")

    # Visualization
    st.subheader('Strategy Execution and Transaction Costs')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df['Time Slice'], df['Execution Volume'],
             label='Execution Volume', color='b', marker='o')
    ax1.set_xlabel('Time Slice')
    ax1.set_ylabel('Execution Volume', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(df['Time Slice'], df['Transaction Cost'],
             label='Transaction Cost', color='r', linestyle='--', marker='x')
    ax2.plot(df['Time Slice'], predicted_costs,
             label='Predicted Cost', color='g', linestyle=':')
    ax2.set_ylabel('Transaction Cost', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(
        0, 1), bbox_transform=ax1.transAxes)
    st.pyplot(fig)

    # Add Explanation of Relation to Chapter 7
    st.markdown("""
    ### Relation to Chapter 7: Machine Learning and Big Data Trade Execution Support
    This simulation app aligns with Chapter 7 by demonstrating:

    1. **Dynamic Strategy Simulation**: The app allows users to simulate **TWAP**, **VWAP**, and **POV** strategies, aligning with the chapter's focus on enhancing execution strategies using ML.
    2. **Transaction Cost Prediction**: A simple linear regression model predicts transaction costs, showcasing the practical application of ML models for cost analysis, as discussed in the chapter.
    3. **Market Adaptation**: The app simulates different market conditions (stable, rising, falling), reflecting the chapter’s emphasis on adapting execution strategies to real-time market changes.
    4. **Interactive Learning**: Users can adjust parameters like trade volume, market impact, and slippage, mirroring the chapter’s approach to hands-on optimization of trade execution.
    """)
