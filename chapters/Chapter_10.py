import streamlit as st
import pandas as pd
import yfinance as yf
from flaml import AutoML
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def chapter10():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # App title and description
    st.subheader("Chapter 10: Accelerated AI in Investment Management")
    st.divider()
    st.write("""
    ### A Portfolio Analysis Example with ESG and Volatility Data
    This Streamlit app demonstrates accelerated AI applications in investment management using ESG (Environmental, Social, Governance) factors and financial metrics, 
    such as returns and volatility. This app is inspired by Chapter 10 of 'Accelerated AI and Use Cases in Investment Management'.
    """)

    # Generate synthetic data
    np.random.seed(42)
    n_assets = 100
    data = pd.DataFrame({
        'Asset_ID': [f'Asset_{i+1}' for i in range(n_assets)],
        'Return': np.random.normal(0.07, 0.02, n_assets),
        'Volatility': np.random.normal(0.15, 0.05, n_assets),
        'ESG_Score': np.random.randint(1, 100, n_assets)
    })

    # Display data
    st.write("### Sample Data")
    st.dataframe(data.head())

    # Sidebar filter options
    st.sidebar.header("Filter Options")
    esg_threshold = st.sidebar.slider("Minimum ESG Score", 1, 100, 50)
    return_threshold = st.sidebar.slider(
        "Minimum Expected Return", 0.00, 0.15, 0.05)
    volatility_max = st.sidebar.slider("Maximum Volatility", 0.05, 0.3, 0.15)

    # Filtered Data
    filtered_data = data[(data['ESG_Score'] >= esg_threshold) &
                         (data['Return'] >= return_threshold) &
                         (data['Volatility'] <= volatility_max)]
    st.write("### Filtered Portfolio")
    st.write(f"Number of assets after filtering: {filtered_data.shape[0]}")
    st.dataframe(filtered_data)

    # Data visualization
    st.write("### Portfolio Visualization")

    # Plot ESG scores vs. returns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_data, x='ESG_Score', y='Return',
                    hue='Volatility', palette="viridis", ax=ax)
    ax.set_title("ESG Scores vs. Returns (Filtered)")
    st.pyplot(fig)

    # Portfolio Analysis
    st.write("### Portfolio Analysis")

    mean_return = filtered_data['Return'].mean()
    mean_volatility = filtered_data['Volatility'].mean()
    st.write(f"**Mean Expected Return:** {mean_return:.2%}")
    st.write(f"**Mean Volatility:** {mean_volatility:.2%}")

    # Conclusion and relevance to Chapter 10
    st.write("""
    ## Conclusion
    This app provides a simplified simulation of how accelerated AI applications can support investment management by enabling large-scale, 
    real-time filtering and visualization of investment data based on ESG factors and financial metrics. In Chapter 10, we explore the advantages of using 
    accelerated computing platforms, such as GPUs, in investment management for tasks like ESG monitoring, risk management, and portfolio analysis. 
    Leveraging these platforms allows investment firms to process vast amounts of data more efficiently, make quicker decisions, and maintain robust 
    and explainable AI models for a sustainable and data-driven approach to investing.
    """)
