import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def chapter8():
    # Title and Introduction
    st.subheader(
        'Chapter 8: ML for Microstructure Data-Driven Execution Algorithms')
    st.markdown("""
    ### Overview
    This app allows you to interact with **limit order book (LOB) data**, visualize **order book imbalance**, and simulate **execution strategies** like TWAP and VWAP.
    """)

    # Sidebar: User inputs
    st.sidebar.header('Settings')

    # Take API Key input from user
    api_key = st.sidebar.text_input(
        'Enter Alpha Vantage API Key', type='password')

    # Prompt to select strategy
    strategy = st.sidebar.selectbox(
        'Select Execution Strategy', ['TWAP', 'VWAP'])
    order_volume = st.sidebar.slider(
        'Order Volume', min_value=100, max_value=10000, step=100, value=1000)

    # Fetching LOB data using Alpha Vantage API (e.g., IBM)
    def fetch_lob_data(symbol='IBM', interval='1min', api_key=None):
        if not api_key:
            st.error("API Key is required to fetch data.")
            return None
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&datatype=csv"
        try:
            df = pd.read_csv(url)
            df['time'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            st.error(f"Error fetching data from Alpha Vantage: {e}")
            return None

    # Load LOB data only if the API key is provided
    if api_key:
        data = fetch_lob_data(api_key=api_key)
    else:
        data = None

    # Display data preview and analysis only if data is available
    if data is not None:
        st.subheader('Real-Time LOB Data')
        st.write(data.head())

        # Feature Engineering: Order Book Imbalance (OBI)
        data['bid_depth'] = data['low']
        data['ask_depth'] = data['high']
        data['OBI'] = (data['bid_depth'] - data['ask_depth']) / \
            (data['bid_depth'] + data['ask_depth'])

        # Plotting Order Book Imbalance
        st.subheader('Order Book Imbalance (OBI)')
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=data, x='time', y='OBI', ax=ax, color='blue')
        ax.set_title('Order Book Imbalance Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('OBI')
        st.pyplot(fig)  # Ensure the plot updates dynamically

        # Simulating Execution Strategies
        st.subheader('Simulated Execution Strategy')
        execution_slices = 10
        slice_size = order_volume // execution_slices

        # Adjust execution times to ensure they are within bounds
        execution_times = np.linspace(
            0, len(data) - 1, execution_slices, dtype=int)
        execution_times = np.clip(execution_times, 0, len(data) - 1)

        if strategy == 'TWAP':
            executed_prices = data['close'].iloc[execution_times]
            executed_volumes = [slice_size] * execution_slices
            st.write(f"TWAP Simulation: {execution_slices} slices of {slice_size} units each.")

        elif strategy == 'VWAP':
            volume_profile = data['volume'].cumsum() / data['volume'].sum()
            execution_times = (volume_profile * (len(data) - 1)).astype(int)
            execution_times = np.clip(execution_times, 0, len(data) - 1)
            executed_prices = data['close'].iloc[execution_times]
            executed_volumes = [slice_size] * execution_slices
            st.write(f"VWAP Simulation: {execution_slices} slices of {slice_size} units each.")

        # Plotting Executed Trades
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['time'], data['close'],
                label='Close Price', color='lightgray')
        ax.scatter(data['time'].iloc[execution_times],
                   executed_prices, color='red', label='Executed Trade', s=50)
        ax.set_title(f'{strategy} Execution Simulation')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)  # Ensure the plot updates dynamically

    else:
        st.warning(
            "Please enter a valid API key to load data. You can get it here: https://www.alphavantage.co/support/#api-key")

    # Conclusion
    st.markdown("""
    ### How This App Relates to Chapter 8
    This application demonstrates the concepts discussed in **Chapter 8** by allowing users to interact with **real-time microstructure data**, visualize **order book imbalances**, and simulate execution strategies like **TWAP** and **VWAP**. This hands-on experience helps in understanding how ML models can optimize trade execution based on LOB dynamics and minimize market impact.
    """)
