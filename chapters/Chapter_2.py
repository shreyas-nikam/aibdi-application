import streamlit as st
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import requests
import feedparser
import plotly.express as px


def chapter2():

    # Title and Introduction
    st.subheader("Chapter 2: Alternative Data and AI in Investment Research")
    st.divider()
    st.markdown("""
    ### Overview
    This app demonstrates the use of **alternative data**, such as **news sentiment from Google News**, to enhance investment decision-making, as discussed in **Chapter 2**. By analyzing news sentiment and correlating it with **stock price movements**, users can gain insights into market behavior. This aligns with the chapter's focus on integrating various data sources for a more comprehensive investment approach.
    """)

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    assets = st.sidebar.multiselect(
        "Select Stock Symbols", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        default=["AAPL", "GOOGL"]
    )
    start_date = st.sidebar.date_input(
        "Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

    # Fetch stock data
    @st.cache_data
    def load_stock_data(assets, start, end):
        try:
            data = yf.download(assets, start=start, end=end)['Adj Close']
            return data
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return None

    # Function to fetch news from Google News RSS Feed
    def fetch_google_news_rss(query, limit=100):
        feed_url = f"https://news.google.com/rss/search?q={
            query}+when:1d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(feed_url)
        news_items = feed.entries[:limit]
        news_data = [{'title': item.title, 'summary': item.summary}
                     for item in news_items]
        return news_data

    # Load stock data
    stock_data = load_stock_data(assets, start_date, end_date)

    if stock_data is not None:
        st.write("### Stock Price Movements")
        st.line_chart(stock_data)

        # News sentiment analysis
        st.subheader("Sentiment Analysis from Google News")
        query = st.text_input(
            "Enter keyword for sentiment analysis (e.g., 'AAPL', 'GOOGL')", "AAPL")
        article_count = st.slider(
            "Number of News Articles to Analyze", 5, 50, 10)

        if st.button("Fetch and Analyze News"):
            with st.spinner("Fetching news and analyzing sentiment..."):
                news_data = fetch_google_news_rss(query, article_count)
                if news_data:
                    news_df = pd.DataFrame(news_data)
                    news_df['Sentiment'] = news_df['summary'].apply(
                        lambda text: TextBlob(text).sentiment.polarity)
                    st.write(news_df[['title', 'Sentiment']])

                    # Sentiment visualization
                    fig = px.histogram(
                        news_df, x="Sentiment", nbins=20, title="News Sentiment Distribution")
                    st.plotly_chart(fig)

                    # Correlation with stock price movements
                    mean_sentiment = news_df['Sentiment'].mean()
                    st.write(f"### Average News Sentiment for '{
                             query}': {mean_sentiment}")

                    st.write("### Correlation Analysis")
                    returns = stock_data.pct_change().dropna()
                    if query.upper() in returns.columns:
                        sentiment_series = pd.Series(
                            [mean_sentiment] * len(returns), index=returns.index)
                        correlation = returns[query.upper()].corr(
                            sentiment_series)
                        st.write(f"Correlation between '{
                                 query}' News Sentiment and Stock Returns: {correlation: .2f}")
                    else:
                        st.warning(
                            "Selected keyword does not match the chosen stock symbols.")
                else:
                    st.warning(
                        "No news articles fetched for the selected keyword.")

    else:
        st.info("Please select assets and date range to load stock data.")

    # How the App Relates to Chapter 2
    st.markdown("""
    ### How This App Relates to Chapter 2
    This app leverages **alternative data**, specifically **news sentiment from Google News**, to provide a comprehensive view of stock price movements:
    - **Integration of Alternative Data**: As described in Chapter 2, this app incorporates non-traditional data sources (e.g., news sentiment) into investment analysis.
    - **Dynamic Correlation**: By correlating news sentiment with stock returns, users gain a clearer picture of how news impacts stock prices.
    - **Real-Time Insights**: Real-time data from news articles provides timely information, aligning with the chapter's emphasis on using alternative data for informed investment decisions.
    """)
