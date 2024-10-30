

def chapter1():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import requests
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the page title
    st.subheader(
        'Chapter 1: Predicting Cross-Sectional Stock Returns Using ML Models')

    st.divider()

    # Sidebar for user input
    st.sidebar.header('User Input Parameters')

    # Sidebar: Stock symbol input
    stock_symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')

    # Sidebar: API key input
    api_key = st.sidebar.text_input('Alpha Vantage API Key', type='password')

    # Sidebar: Model selection
    model_option = st.sidebar.selectbox(
        'Select Model',
        ('Random Forest', 'Gradient Boosting', 'Neural Network')
    )

    # Check if inputs are provided
    if not api_key:
        st.warning(
            'Please enter a valid API key in the sidebar. Get it from here: https://www.alphavantage.co/support/#api-key')
    elif not stock_symbol:
        st.warning('Please enter a valid stock symbol in the sidebar.')
    else:
        # Fetch data button
        if st.sidebar.button('Fetch and Analyze Data'):
            with st.spinner('Fetching stock data...'):
                # Fetch stock data from Alpha Vantage API
                def fetch_stock_data(symbol, api_key, interval='daily'):
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_{
                        interval.upper()}&symbol={symbol}&apikey={api_key}&outputsize=full"
                    response = requests.get(url)
                    data = response.json()
                    if "Time Series (Daily)" in data:
                        df = pd.DataFrame(data['Time Series (Daily)']).T
                        df.columns = ['open', 'high', 'low', 'close', 'volume']
                        df = df.astype(float)
                        return df
                    else:
                        return None

                # Fetch data
                stock_data = fetch_stock_data(stock_symbol, api_key)

                if stock_data is None:
                    st.error(
                        "Error fetching data. Please check your API key and symbol.")
                else:
                    st.success(f"Stock data for {
                               stock_symbol} fetched successfully!")

                    # Calculate returns
                    stock_data['return'] = stock_data['close'].pct_change()
                    stock_data.dropna(inplace=True)

                    # Display the raw data
                    st.subheader('Stock Data (Last 5 Days)')
                    st.write(stock_data.tail())

                    # Plot stock closing prices
                    with st.spinner('Generating closing price plot...'):
                        st.subheader('Stock Closing Price Over Time')
                        plt.figure(figsize=(8, 4))
                        plt.plot(stock_data.index,
                                 stock_data['close'], label='Closing Price')
                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.title(
                            f'{stock_symbol} Stock Closing Price Over Time')
                        plt.legend()
                        st.pyplot(plt)
                        st.markdown("""
                        **Explanation:** This graph shows the historical trend of the closing prices of the selected stock. 
                        It provides insights into the stock's performance over time and helps visualize potential upward or downward trends.
                        """)

                    # Plot the distribution of returns
                    with st.spinner('Generating return distribution plot...'):
                        st.subheader('Distribution of Stock Returns')
                        plt.figure(figsize=(8, 4))
                        sns.histplot(
                            stock_data['return'], bins=50, kde=True, color='skyblue')
                        plt.xlabel('Return')
                        plt.title('Distribution of Stock Returns')
                        st.pyplot(plt)
                        st.markdown("""
                        **Explanation:** This histogram shows the distribution of the stock's daily returns. 
                        It helps identify the frequency and spread of returns, giving an idea of volatility and potential risk.
                        """)

                    # Feature engineering
                    stock_data['ma_10'] = stock_data['close'].rolling(
                        window=10).mean()
                    stock_data['volatility_10'] = stock_data['return'].rolling(
                        window=10).std()
                    stock_data['lagged_return'] = stock_data['return'].shift(1)

                    # Drop NaN values
                    stock_data.dropna(inplace=True)

                    # Define features and target
                    features = ['ma_10', 'volatility_10', 'lagged_return']
                    X = stock_data[features]
                    y = stock_data['return']

                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)

                    # Model training
                    st.subheader('Model Training and Evaluation')

                    with st.spinner('Training the model...'):
                        if model_option == 'Random Forest':
                            model = RandomForestRegressor(
                                n_estimators=100, random_state=42)
                        elif model_option == 'Gradient Boosting':
                            model = GradientBoostingRegressor(
                                n_estimators=100, random_state=42)
                        elif model_option == 'Neural Network':
                            model = MLPRegressor(hidden_layer_sizes=(
                                50, 50), max_iter=500, random_state=42)

                        # Train model
                        model.fit(X_train, y_train)

                        # Make predictions
                        preds = model.predict(X_test)

                        # Evaluate model
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        r2 = r2_score(y_test, preds)

                        # Display evaluation metrics
                        st.write(f'**Model Selected:** {model_option}')
                        st.write(f'**RMSE:** {rmse:.4f}')
                        st.write(f'**R2 Score:** {r2:.4f}')

                    # Plot actual vs. predicted returns
                    with st.spinner('Generating actual vs. predicted plot...'):
                        st.subheader('Actual vs. Predicted Returns')
                        plt.figure(figsize=(6, 4))
                        plt.scatter(y_test, preds, alpha=0.6)
                        plt.xlabel('Actual Returns')
                        plt.ylabel('Predicted Returns')
                        plt.title(
                            f'{model_option}: Actual vs. Predicted Returns')
                        st.pyplot(plt)
                        st.markdown("""
                        **Explanation:** This scatter plot compares the actual vs. predicted returns for the selected model. 
                        It provides a visual sense of the model's accuracy. Ideally, points should be closer to the diagonal line, indicating better predictions.
                        """)

                    # Feature importance for Random Forest and Gradient Boosting
                    if model_option in ['Random Forest', 'Gradient Boosting']:
                        with st.spinner('Generating feature importance plot...'):
                            st.subheader('Feature Importance')
                            feature_importance = pd.Series(
                                model.feature_importances_, index=features)
                            plt.figure(figsize=(6, 3))
                            feature_importance.sort_values(
                                ascending=False).plot(kind='bar', color='purple')
                            plt.xlabel('Features')
                            plt.ylabel('Importance')
                            plt.title(f'Feature Importance ({model_option})')
                            st.pyplot(plt)
                            st.markdown("""
                            **Explanation:** This bar plot shows the relative importance of each feature in the model. 
                            It helps identify which factors are most influential in predicting the stock's returns, aiding in better decision-making for stock selection.
                            """)

    # Sidebar: Add footer information
    st.sidebar.markdown("""
    ---
    *Created as part of Chapter 1: ML Applications in Investments Course*
    """)
