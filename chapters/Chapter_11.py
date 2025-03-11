import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def chapter11():

    # Title and Introduction
    st.subheader("Chapter 11: Symbolic AI: A Case Study")
    st.markdown("""
    ### Overview
    This interactive app demonstrates **Symbolic AI** in portfolio management by automating tasks 
    like **rebalancing** and **risk assessment** using rule-based logic. In this version, the app 
    uses *synthetic* financial data rather than real-time data from yfinance, but the workflow 
    remains identical to the original example.
    """)

    # Sidebar for user input
    st.sidebar.header("User Inputs")
    assets = st.sidebar.multiselect(
        "Select Portfolio Assets",
        ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        default=["AAPL", "GOOGL", "MSFT"]
    )
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

    # Allow users to set target weights for rebalancing
    st.sidebar.subheader("Set Target Weights")
    target_weights = {
        asset: st.sidebar.slider(f"{asset} Target Weight", 0.0, 1.0, 0.2, 0.05) 
        for asset in assets
    }

    # Ensure total target weights sum to 1
    total_weight = sum(target_weights.values())
    if total_weight != 0:  # Avoid division by zero if user sets everything to 0
        target_weights = {
            k: v / total_weight for k, v in target_weights.items()
        }

    # Generate synthetic financial data
    @st.cache_data
    def generate_synthetic_data(assets, start, end):
        """
        Creates synthetic daily 'Adj Close' price data for the selected assets,
        simulating random-walk-like behavior over the given date range.
        """
        dates = pd.date_range(start=start, end=end, freq='B')  # business days
        synthetic_df = pd.DataFrame(index=dates)

        for asset in assets:
            # Start price between 50 and 300
            initial_price = np.random.uniform(50, 300)
            # Generate random daily returns from a normal distribution
            daily_returns = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))

            prices = [initial_price]
            for ret in daily_returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            synthetic_df[asset] = prices
        
        return synthetic_df

    # Load (generate) synthetic data
    data = None
    if assets:
        data = generate_synthetic_data(assets, start_date, end_date)

    if data is not None and not data.empty:
        st.write("### Asset Prices (Synthetic Data)")
        st.line_chart(data)

        # Calculate daily returns
        returns = data.pct_change().dropna()

        # Define initial portfolio with equal weights
        if len(assets) > 0:
            initial_weights = {asset: 1 / len(assets) for asset in assets}
        else:
            initial_weights = {}

        # Rebalancing rule
        def rebalance_portfolio(weights, target_weights):
            """
            Rebalances the portfolio to match target weights.
            """
            rebalanced_weights = target_weights
            return rebalanced_weights

        # Apply rebalancing
        rebalanced_weights = rebalance_portfolio(initial_weights, target_weights)

        # Allow users to set a custom risk threshold
        st.sidebar.subheader("Set Risk Threshold")
        risk_threshold = st.sidebar.slider(
            "Risk Threshold", 0.0, 0.1, 0.02, 0.01
        )

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

        # Filter assets by risk category if desired
        selected_risk_category = st.sidebar.selectbox(
            "Filter by Risk Category", ["All", "High Risk", "Low Risk"]
        )
        if selected_risk_category != "All":
            filtered_assets = [
                asset for asset, risk in risk_assessment.items()
                if risk == selected_risk_category
            ]
            # Filter returns DataFrame to include only selected assets
            returns = returns[filtered_assets]

        # Combine weights and risk assessment into a single DataFrame
        combined_data = pd.DataFrame({
            'Asset': initial_weights.keys(),
            'Initial Weight': initial_weights.values(),
            'Target Weight': target_weights.values(),
            'Rebalanced Weight': rebalanced_weights.values(),
            'Risk Category': [risk_assessment[asset] for asset in initial_weights.keys()]
        })

        # Define functions to apply color coding
        def highlight_weights(val):
            # Highlight weights >= 0.3 in green, others in lightcoral
            color = 'green' if val >= 0.3 else 'lightcoral'
            return f'background-color: {color}'

        def highlight_risk(val):
            # Highlight high risk in red, low risk in darkgreen
            if val == 'High Risk':
                color = 'red'
            elif val == 'Low Risk':
                color = 'darkgreen'
            else:
                color = ''
            return f'background-color: {color}'

        # Apply color coding to the DataFrame
        styled_combined_data = combined_data.style.applymap(
            highlight_weights,
            subset=['Initial Weight', 'Target Weight', 'Rebalanced Weight']
        ).applymap(
            highlight_risk,
            subset=['Risk Category']
        )

        # Display the color-coded table
        st.write("### Portfolio Overview")
        st.dataframe(styled_combined_data, use_container_width=True)

        # Visualization of initial vs. rebalanced weights
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            initial_weights.keys(),
            initial_weights.values(),
            alpha=0.6,
            label='Initial Weights'
        )
        ax.bar(
            rebalanced_weights.keys(),
            rebalanced_weights.values(),
            alpha=0.6,
            label='Rebalanced Weights'
        )
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weights')
        ax.set_title('Initial vs. Rebalanced Portfolio Weights')
        ax.legend()
        st.pyplot(fig)

        # Visualization of risk categories
        risk_df = pd.DataFrame(list(risk_assessment.items()), columns=['Asset', 'Risk Category'])
        st.write("### Risk Categories")
        st.bar_chart(risk_df.set_index('Asset'))

    else:
        st.info("Please select assets and date range to generate synthetic data.")

    # How the App Relates to Chapter 11
    st.markdown("""
    ### How This App Relates to Chapter 11
    This interactive app aligns with **Chapter 11** by demonstrating symbolic AI principles in 
    portfolio management:
    - **Rule-Based Rebalancing**: It uses predefined rules to rebalance asset weights based on 
      user inputs, showcasing how symbolic AI provides transparent decision-making.
    - **Dynamic Risk Assessment**: Allows users to customize risk thresholds, reflecting 
      symbolic AI's interpretability and adaptability.
    - **Synthetic Data Simulation**: Here, we simulate asset prices rather than using real-time 
      data. The rule-based logic remains the same, demonstrating how symbolic AI can be applied 
      regardless of data source.
    """)

# To run as a standalone Streamlit app, just call chapter11() in your main script:
# if __name__ == "__main__":
#     import streamlit as st
#     chapter11()
