import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import feedparser
import plotly.express as px

def chapter2():
    # Title and Introduction
    st.subheader("Chapter 2: Alternative Data and AI in Investment Research")
    st.divider()
    st.markdown("""
    ### Overview
    This app demonstrates the use of **alternative data**, such as **news sentiment from Google News**, 
    to enhance investment decision-making, as discussed in **Chapter 2**. By analyzing news sentiment 
    and correlating it with **stock price movements** (simulated here), users can gain insights into 
    market behavior. This aligns with the chapter's focus on integrating various data sources for a 
    more comprehensive investment approach.
    """)

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    assets = st.sidebar.multiselect(
        "Select Stock Symbols", 
        ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"], 
        default=["AAPL", "GOOGL"]
    )
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

    # Generate synthetic stock data
    @st.cache_data
    def generate_synthetic_stock_data(assets, start, end):
        """
        Create synthetic time series data for chosen assets between start and end dates.
        Each asset simulates daily 'Adj Close' prices as a random walk.
        """
        dates = pd.date_range(start=start, end=end, freq='B')  # business days
        data = pd.DataFrame(index=dates)

        for asset in assets:
            # Start each asset at a random initial price between 50 and 300
            initial_price = np.random.uniform(50, 300)
            
            # Generate daily returns as random draws from a normal distribution
            daily_returns = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))
            
            # Create a price series by cumulatively applying returns to the initial price
            price_series = [initial_price]
            for ret in daily_returns[1:]:
                new_price = price_series[-1] * (1 + ret)
                price_series.append(new_price)

            data[asset] = price_series
        
        return data

    stock_data = None
    if assets:
        stock_data = generate_synthetic_stock_data(assets, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        st.write("### Simulated Stock Price Movements")
        st.line_chart(stock_data)

        # News sentiment analysis
        st.subheader("Sentiment Analysis from Google News")
        query = st.text_input(
            "Enter keyword for sentiment analysis (e.g., 'AAPL', 'GOOGL')", 
            "AAPL"
        )
        article_count = st.slider(
            "Number of News Articles to Analyze", 
            min_value=5, 
            max_value=50, 
            value=10
        )

        # Function to fetch news from Google News RSS Feed
        def fetch_google_news_rss(query, limit=100):
            """
            Fetches articles from Google News RSS feed for the given query.
            """
            feed_url = f"https://news.google.com/rss/search?q={query}+when:1d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(feed_url)
            news_items = feed.entries[:limit]
            news_data = [{'title': item.title, 'summary': item.summary} for item in news_items]
            return news_data

        if st.button("Fetch and Analyze News"):
            with st.spinner("Fetching news and analyzing sentiment..."):
                news_data = fetch_google_news_rss(query, article_count)
                if news_data:
                    news_df = pd.DataFrame(news_data)
                    news_df['Sentiment'] = news_df['summary'].apply(
                        lambda text: TextBlob(text).sentiment.polarity
                    )
                    st.write(news_df[['title', 'Sentiment']])

                    # Sentiment visualization
                    fig = px.histogram(
                        news_df, 
                        x="Sentiment", 
                        nbins=20, 
                        title="News Sentiment Distribution"
                    )
                    st.plotly_chart(fig)

                    # Correlation with stock price movements
                    mean_sentiment = news_df['Sentiment'].mean()
                    st.write(f"### Average News Sentiment for '{query}': {mean_sentiment:.2f}")

                    st.write("### Correlation Analysis")
                    returns = stock_data.pct_change().dropna()
                    
                    # Check if the query matches any of the user-selected assets
                    if query.upper() in returns.columns:
                        sentiment_series = pd.Series(
                            [mean_sentiment] * len(returns), 
                            index=returns.index
                        )
                        correlation = returns[query.upper()].corr(sentiment_series)
                        st.write(
                            f"Correlation between '{query}' News Sentiment and Stock Returns: {correlation:.2e}"
                        )
                    else:
                        st.warning("Selected keyword does not match the chosen stock symbols.")
                else:
                    st.warning("No news articles fetched for the selected keyword.")
    else:
        st.info("Please select assets and date range to generate synthetic stock data.")

    # How the App Relates to Chapter 2
    st.markdown("""
    ### How This App Relates to Chapter 2
    This app leverages **alternative data**, specifically **news sentiment from Google News**, 
    to provide a comprehensive view of stock price movements (simulated here):
    - **Integration of Alternative Data**: As described in Chapter 2, this app incorporates 
      non-traditional data sources (e.g., news sentiment) into investment analysis.
    - **Dynamic Correlation**: By correlating news sentiment with (simulated) stock returns, 
      users gain a clearer picture of how news might impact stock prices.
    - **Real-Time Insights**: Real-time data from news articles provides timely information, 
      aligning with the chapter's emphasis on using alternative data for informed investment decisions.
    """)

# To run this as a standalone Streamlit app, simply call chapter2() in your main script:
# if __name__ == "__main__":
#     import streamlit as st
#     chapter2()
