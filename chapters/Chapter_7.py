import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def chapter7():
    # App Title
    st.subheader(
        'Chapter 7: Machine Learning and Big Data Trade Execution Support')
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
                                         'TWAP', 'VWAP', 'POV (Proportion of Volume)'])
    market_impact = st.sidebar.slider(
        'Market Impact Factor', min_value=0.0, max_value=0.1, value=0.05, step=0.01)
    slippage = st.sidebar.slider(
        'Slippage Factor', min_value=0.0, max_value=0.05, value=0.01, step=0.01)

    # Simulate Market Conditions
    st.sidebar.subheader('Simulated Market Conditions')
    market_condition = st.sidebar.selectbox(
        'Market Condition', ['Stable', 'Rising', 'Falling'])
    execution_period = st.sidebar.slider(
        'Execution Period (Slices)', min_value=30, max_value=100, value=50)

    # Set different random seeds for each market condition for variation
    if market_condition == 'Stable':
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.002, execution_period)
        volume_mean = 8.5
    elif market_condition == 'Rising':
        np.random.seed(24)
        price_changes = np.random.normal(0.002, 0.003, execution_period)  # Stronger upward drift
        volume_mean = 8.7
    else:  # Falling
        np.random.seed(84)
        price_changes = np.random.normal(-0.002, 0.003, execution_period)  # Stronger downward drift
        volume_mean = 8.3

    # Generate market prices with a random walk model
    initial_price = 100
    market_prices = initial_price * np.cumprod(1 + price_changes)

    # Generate market volumes with a log-normal distribution based on the mean
    market_volumes = np.random.lognormal(mean=volume_mean, sigma=0.2, size=execution_period)

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

    # Execution Volume Plot
    ax1.plot(df['Time Slice'], df['Execution Volume'],
             label='Execution Volume', color='blue', marker='o', linestyle='-')
    ax1.set_xlabel('Time Slice')
    ax1.set_ylabel('Execution Volume', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Transaction Cost Plot on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['Time Slice'], df['Transaction Cost'],
             label='Transaction Cost', color='red', linestyle='--', marker='x')
    ax2.set_ylabel('Transaction Cost', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Predicted Cost Plot on a separate secondary y-axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    ax3.plot(df['Time Slice'], predicted_costs,
             label='Predicted Cost', color='green', linestyle=':', marker='s')
    ax3.set_ylabel('Predicted Cost', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Add legends for all plots
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
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

