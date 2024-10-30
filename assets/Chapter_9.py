import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import pandas as pd
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize


# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')


def chapter9():

    # Title and Introduction
    st.title("Customer Feedback Insights for Finance")
    st.markdown("""
    ### Overview
    This application provides **insights into customer feedback** using NLP techniques. It helps financial institutions identify key themes, sentiment trends, and frequently mentioned keywords to enhance customer service strategies.
    * It aligns with the learnings from **Chapter 9**, which emphasizes the role of AI in understanding customer needs and improving service efficiency.
    """)

    # Sidebar for File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file with a 'feedback' column", type=["csv"])
    use_sample = st.sidebar.button("Use Sample File Instead")

    # Load Sample Data if selected
    if use_sample:
        uploaded_file = 'data/sample_customer_feedback.csv'
        st.success("Using the sample file provided.")

    # Load and Display Data
    if uploaded_file:
        try:
            # Load data from the uploaded file or sample file
            feedback_data = pd.read_csv(uploaded_file)

            # Check if 'feedback' column exists
            if 'feedback' not in feedback_data.columns:
                st.error("The uploaded file must have a 'feedback' column.")
            else:
                st.write("### Customer Feedback Data")
                st.dataframe(feedback_data.head())

                # Data Preprocessing
                feedback_data['processed_feedback'] = feedback_data['feedback'].apply(
                    lambda x: ' '.join(
                        [word for word in word_tokenize(str(x).lower()) if word.isalnum()])
                )

                # Sentiment Analysis
                sentiment_analyzer = pipeline("sentiment-analysis")

                st.write("### Sentiment Distribution")
                feedback_data['sentiment'] = feedback_data['feedback'].apply(
                    lambda x: sentiment_analyzer(str(x))[0]['label'])

                # Plot sentiment distribution
                sentiment_counts = feedback_data['sentiment'].value_counts()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title("Sentiment Distribution in Customer Feedback")
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Count")
                st.pyplot(fig)

                # Topic Modeling (Keyword Extraction)
                st.write("### Keyword Extraction")
                vectorizer = CountVectorizer(
                    max_features=50, stop_words='english')
                X = vectorizer.fit_transform(
                    feedback_data['processed_feedback'])
                keywords = vectorizer.get_feature_names_out()

                # Display keywords as a word cloud
                wordcloud = WordCloud(
                    width=800, height=400, background_color='white').generate(' '.join(keywords))
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

                # Key Themes Extraction
                st.write("### Key Themes in Customer Feedback")
                top_keywords = pd.DataFrame({'Keyword': keywords})
                st.dataframe(top_keywords)

        except Exception as e:
            st.error(f"Error loading the file: {e}")

    else:
        st.info("Please upload a CSV file or use the sample file provided.")

    # Conclusion and Relation to Chapter 9
    st.markdown("""
    ### How This App Relates to Chapter 9
    This app demonstrates how AI can be used to analyze customer feedback, providing valuable insights into sentiment trends, key themes, and frequently mentioned topics.
    It aligns with **Chapter 9** by showcasing AI's role in transforming financial customer service beyond direct interactions to strategic insights, helping organizations optimize services and address customer needs more effectively.
    """)
