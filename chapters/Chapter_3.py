import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def chapter3():
    # App Title
    st.subheader('Chapter 3: Data Science for Active and Long-Term Fundamental Investing')
    st.divider()
    st.markdown("""
    This app allows you to simulate different scenarios for long-term financial indicators and visualize their potential impact on long-term returns. You can also compare multiple scenarios side by side.""")

    # Sidebar Inputs
    st.sidebar.header('Scenario Inputs')

    # Number of scenarios to compare
    num_scenarios = st.sidebar.selectbox(
        'Number of Scenarios to Compare', [1, 2, 3], index=0)

    # Initialize user input storage
    scenarios = []

    # Loop through each scenario input panel
    for i in range(num_scenarios):
        st.sidebar.subheader(f'Scenario {i + 1}')

        # User inputs for financial indicators
        earnings_growth = st.sidebar.slider(
            f'Scenario {i + 1} - Earnings Growth (%)', -10, 30, 5, 1)
        pe_ratio = st.sidebar.slider(
            f'Scenario {i + 1} - P/E Ratio', 5, 40, 15, 1)
        pb_ratio = st.sidebar.slider(
            f'Scenario {i + 1} - P/B Ratio', 0.5, 10.0, 2.0, 0.1)
        dividend_yield = st.sidebar.slider(
            f'Scenario {i + 1} - Dividend Yield (%)', 0, 10, 3, 1)
        debt_to_equity = st.sidebar.slider(
            f'Scenario {i + 1} - Debt-to-Equity Ratio', 0.0, 3.0, 1.0, 0.1)

        # Store the scenario inputs
        scenarios.append({
            'Earnings Growth': earnings_growth,
            'P/E Ratio': pe_ratio,
            'P/B Ratio': pb_ratio,
            'Dividend Yield': dividend_yield,
            'Debt-to-Equity': debt_to_equity
        })

    # Generate synthetic data for simulation
    np.random.seed(42)

    n_points = 200  # Number of data points

    # Create synthetic data for each indicator
    earnings_growth_data = np.random.uniform(0, 20, n_points)
    pe_ratio_data = np.random.uniform(5, 40, n_points)
    pb_ratio_data = np.random.uniform(0.5, 10.0, n_points)
    dividend_yield_data = np.random.uniform(0, 10, n_points)
    debt_to_equity_data = np.random.uniform(0, 3, n_points)

    # Simulated returns based on indicators
    returns_data = (
        0.3 * earnings_growth_data +
        0.2 * pe_ratio_data +
        0.2 * pb_ratio_data +
        0.15 * dividend_yield_data -
        0.15 * debt_to_equity_data
    ) / 100 + np.random.normal(0, 0.01, n_points)

    # Create a DataFrame for the simulation
    data = pd.DataFrame({
        'Earnings Growth': earnings_growth_data,
        'P/E Ratio': pe_ratio_data,
        'P/B Ratio': pb_ratio_data,
        'Dividend Yield': dividend_yield_data,
        'Debt-to-Equity': debt_to_equity_data,
        'Long-Term Return': returns_data
    })

    # Train a linear regression model on the synthetic data
    X = data[['Earnings Growth', 'P/E Ratio',
              'P/B Ratio', 'Dividend Yield', 'Debt-to-Equity']]
    y = data['Long-Term Return']

    model = LinearRegression()
    model.fit(X, y)

    # Simulate and visualize scenarios
    st.subheader('Simulated Scenarios')

    # Plot container
    fig, axes = plt.subplots(1, num_scenarios, figsize=(8 * num_scenarios, 6))

    if num_scenarios == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Loop through each scenario for prediction and visualization
    for i, scenario in enumerate(scenarios):
        # Prepare input for prediction
        user_input = np.array([[scenario['Earnings Growth'], scenario['P/E Ratio'],
                              scenario['P/B Ratio'], scenario['Dividend Yield'], scenario['Debt-to-Equity']]])

        # Predict long-term returns for the scenario
        predicted_return = model.predict(user_input)[0]

        # Display scenario inputs
        st.markdown(f"""
        ### Scenario {i + 1} Results
        - **Earnings Growth**: {scenario['Earnings Growth']}%
        - **P/E Ratio**: {scenario['P/E Ratio']}
        - **P/B Ratio**: {scenario['P/B Ratio']}
        - **Dividend Yield**: {scenario['Dividend Yield']}%
        - **Debt-to-Equity Ratio**: {scenario['Debt-to-Equity']}
        """)
        st.write(
            f"**Predicted Long-Term Return**: {predicted_return * 100:.2f}%")

        # Visualize simulated distribution of returns for this scenario
        simulated_returns = (
            0.3 * scenario['Earnings Growth'] +
            0.2 * scenario['P/E Ratio'] +
            0.2 * scenario['P/B Ratio'] +
            0.15 * scenario['Dividend Yield'] -
            0.15 * scenario['Debt-to-Equity']
        ) / 100 + np.random.normal(0, 0.01, n_points)

        sns.histplot(simulated_returns, bins=30, kde=True,
                     ax=axes[i], color='skyblue')
        axes[i].set_title(
            f'Scenario {i + 1}: Distribution of Predicted Returns')
        axes[i].set_xlabel('Simulated Long-Term Return (%)')
        axes[i].set_ylabel('Frequency')

    # Display the plot
    st.pyplot(fig)

    # Additional interaction
    st.markdown("""
    ### Insights
    - **Distribution of Returns**: The histogram shows the distribution of simulated long-term returns based on the selected scenario.
    - **Scenario Comparison**: Adjust the financial indicators in the sidebar to compare different scenarios side by side.
    - **Interpreting Results**: Higher predicted returns generally indicate more favorable scenarios for long-term investment returns, but actual results depend on market conditions and other factors.
    """)

    # Sidebar: Reset button
    if st.sidebar.button('Reset Scenarios'):
        st.experimental_rerun()

    st.markdown("""
                 ### Relation to Chapter 3
    This simulation is directly related to **Chapter 3** of the course, which emphasizes building predictive models for long-term alpha generation using a combination of fundamental financial indicators and macroeconomic factors. In Chapter 3, you learn about integrating data sources, processing long-term data series, and leveraging them to build scalable models that can be used for investment decisions over extended time horizons.

    The app aligns with the chapter’s objectives by:
    1. **Allowing scenario testing** with different financial indicators to understand their impact on long-term returns.
    2. **Facilitating data-driven decision-making** by showing how key financial metrics influence returns.
    3. **Providing an interactive tool** that reflects the principles of long-term modeling discussed in Chapter 3.
    """)
