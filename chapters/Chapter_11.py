import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def chapter11():

    # Title and Introduction
    st.subheader(
        "Chapter 11: Symbolic AI: A Case Study")
    st.markdown("""
    ### Overview
    This interactive app demonstrates **Symbolic AI** in portfolio management by automating tasks like **rebalancing** and **risk assessment** using rule-based logic. It uses real-time financial data and provides dynamic, interactive controls for adjusting portfolio weights and risk thresholds.
    """)

    # Sidebar for user input
    st.sidebar.header("User Inputs")
    assets = st.sidebar.multiselect(
        "Select Portfolio Assets",
        ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        default=["AAPL", "GOOGL", "MSFT"]
    )
    start_date = st.sidebar.date_input(
        "Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

    # Allow users to set target weights for rebalancing
    st.sidebar.subheader("Set Target Weights")
    target_weights = {asset: st.sidebar.slider(
        f"{asset} Target Weight", 0.0, 1.0, 0.2, 0.05) for asset in assets}

    # Adjust total target weights to sum to 1
    total_weight = sum(target_weights.values())
    if total_weight != 1:
        target_weights = {k: v / total_weight for k,
                          v in target_weights.items()}

    # Fetch financial data
    @st.cache_data
    def load_data(assets, start, end):
        try:
            data = yf.download(assets, start=start, end=end)['Adj Close']
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    # Load data
    data = load_data(assets, start_date, end_date)

    if data is not None:
        st.write("### Asset Prices")
        st.line_chart(data)

        # Calculate daily returns
        returns = data.pct_change().dropna()

        # Define initial portfolio with equal weights
        initial_weights = {asset: 1/len(assets) for asset in assets}

        # Rebalancing rule

        def rebalance_portfolio(weights, target_weights):
            """
            Rebalances the portfolio to match target weights.
            """
            rebalanced_weights = target_weights
            return rebalanced_weights

        # Apply rebalancing
        rebalanced_weights = rebalance_portfolio(
            initial_weights, target_weights)

        # Allow users to set a custom risk threshold
        st.sidebar.subheader("Set Risk Threshold")
        risk_threshold = st.sidebar.slider(
            "Risk Threshold", 0.0, 0.1, 0.02, 0.01)

        # Calculate volatility for risk assessment
        volatility = returns.std()

        # Risk assessment rule
        def assess_risk(volatility, threshold):
            """
            Classifies assets as 'High Risk' or 'Low Risk' based on volatility.
            """
            risk_categories = {}
            for asset, vol in volatility.items():
                risk_categories[asset] = 'High Risk' if vol > threshold else 'Low Risk'
            return risk_categories

        # Apply risk assessment
        risk_assessment = assess_risk(volatility, risk_threshold)

        # Filter assets by risk category
        selected_risk_category = st.sidebar.selectbox(
            "Filter by Risk Category", ["All", "High Risk", "Low Risk"])
        if selected_risk_category != "All":
            filtered_assets = [asset for asset, risk in risk_assessment.items(
            ) if risk == selected_risk_category]
            returns = returns[filtered_assets]

        # Combine weights and risk assessment into a single DataFrame
        combined_data = pd.DataFrame({
            'Asset': initial_weights.keys(),
            'Initial Weight': initial_weights.values(),
            'Target Weight': target_weights.values(),
            'Rebalanced Weight': rebalanced_weights.values(),
            'Risk Category': [risk_assessment[asset] for asset in initial_weights.keys()]
        })

        # Define a function to apply color coding
        def highlight_weights(val):
            color = 'green' if val > 0.3 else 'lightcoral'
            return f'background-color: {color}'

        def highlight_risk(val):
            if val == 'High Risk':
                color = 'red'
            elif val == 'Low Risk':
                color = 'darkgreen'
            else:
                color = ''
            return f'background-color: {color}'

        # Apply color coding to the DataFrame
        styled_combined_data = combined_data.style.applymap(
            highlight_weights, subset=[
                'Initial Weight', 'Target Weight', 'Rebalanced Weight']
        ).applymap(
            highlight_risk, subset=['Risk Category']
        )

        # Display the color-coded table in Streamlit
        st.write("### Portfolio Overview")
        st.dataframe(styled_combined_data, use_container_width=True)

        # Visualization of rebalanced weights
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(initial_weights.keys(), initial_weights.values(),
               alpha=0.6, label='Initial Weights', color='blue')
        ax.bar(rebalanced_weights.keys(), rebalanced_weights.values(),
               alpha=0.6, label='Rebalanced Weights', color='orange')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weights')
        ax.set_title('Initial vs. Rebalanced Portfolio Weights')
        ax.legend()
        st.pyplot(fig)

        # Visualization of risk categories
        risk_df = pd.DataFrame(list(risk_assessment.items()), columns=[
                               'Asset', 'Risk Category'])
        st.write("### Risk Categories")
        st.bar_chart(risk_df.set_index('Asset'))

    else:
        st.info("Please select assets and date range to load data.")

    # How the App Relates to Chapter 11
    st.markdown("""
    ### How This App Relates to Chapter 11
    This interactive app aligns with **Chapter 11** by demonstrating symbolic AI principles in portfolio management:
    - **Rule-Based Rebalancing**: It uses predefined rules to rebalance asset weights based on user inputs, showcasing how symbolic AI provides transparent decision-making.
    - **Dynamic Risk Assessment**: Allows users to customize risk thresholds, reflecting symbolic AI’s interpretability and adaptability.
    - **Real-World Data Integration**: Uses real-time market data to provide contextually relevant insights, demonstrating the practical application of symbolic AI in finance.
    """)
