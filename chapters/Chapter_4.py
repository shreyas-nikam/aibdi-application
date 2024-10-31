import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy.cli import download
from pathlib import Path
import subprocess

def chapter4():


    # Define a local directory to install the model
    model_path = Path("models/en_core_web_sm")

    # Check if the model is already available locally
    if not model_path.exists():
        # Download and link the model to the local directory
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm", "--target", "models"])

    # Load the model from the local path
    nlp = spacy.load(str(model_path))
    # Set Streamlit page configuration

    # Title and Introduction
    st.subheader(
        'Chapter 4: Unlocking Insights and Opportunities with NLP in Asset Management')
    st.divider()
    st.markdown("""
    This app lets you input financial news text to analyze sentiment, identify entities, and visualize results using word clouds.
    It aligns with **Chapter 4: Unlocking Insights and Opportunities with NLP in Asset Management**, where NLP is used to extract insights from financial news for sentiment-driven strategies.
    """)

    # User Input
    st.subheader('Input Financial News Text')
    user_input = st.text_area(
        'Enter financial news or headlines here.', height=150)

    if user_input:
        # Sentiment Analysis using TextBlob
        blob = TextBlob(user_input)
        textblob_sentiment = blob.sentiment.polarity

        # Sentiment Analysis using VADER
        vader_analyzer = SentimentIntensityAnalyzer()
        vader_sentiment = vader_analyzer.polarity_scores(user_input)[
            'compound']

        # Entity Recognition using spaCy
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Classify entities based on sentiment
        positive_entities = []
        negative_entities = []
        neutral_entities = []

        for ent, label in entities:
            sentiment = vader_analyzer.polarity_scores(ent)['compound']
            if sentiment > 0.05:
                positive_entities.append(ent)
            elif sentiment < -0.05:
                negative_entities.append(ent)
            else:
                neutral_entities.append(ent)

        # Display Results
        st.subheader('Analysis Results')

        # Display sentiment scores
        st.markdown("### Sentiment Analysis")
        st.write(f"**TextBlob Sentiment Polarity**: {textblob_sentiment}")
        st.write(f"**VADER Sentiment Compound Score**: {vader_sentiment}")

        # Sentiment Classification
        if vader_sentiment > 0.05:
            sentiment_label = ':green[Positive]'
        elif vader_sentiment < -0.05:
            sentiment_label = ':red[Negative]'
        else:
            sentiment_label = ':yellow[Neutral]'

        st.markdown(f"**Overall Sentiment**: {sentiment_label}")

        # Display entity recognition
        st.markdown("### Entity Recognition")
        if entities:
            for entity, label in entities:
                st.write(f"**Entity**: {entity}, **Type**: {label}")
        else:
            st.write("No entities detected.")

        # WordCloud Generation
        st.subheader('Word Clouds for Entities')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Positive Entities")
            if positive_entities:
                positive_wordcloud = WordCloud(
                    width=400, height=300, background_color='white').generate(' '.join(positive_entities))
                plt.figure(figsize=(5, 3))
                plt.imshow(positive_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No positive entities detected.")

        with col2:
            st.markdown("#### Neutral Entities")
            if neutral_entities:
                neutral_wordcloud = WordCloud(
                    width=400, height=300, background_color='white').generate(' '.join(neutral_entities))
                plt.figure(figsize=(5, 3))
                plt.imshow(neutral_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No neutral entities detected.")

        with col3:
            st.markdown("#### Negative Entities")
            if negative_entities:
                negative_wordcloud = WordCloud(
                    width=400, height=300, background_color='white').generate(' '.join(negative_entities))
                plt.figure(figsize=(5, 3))
                plt.imshow(negative_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No negative entities detected.")

    # Add explanation about Chapter 4 alignment
    st.markdown("""
    ### Relation to Chapter 4
    This app reflects the learnings from **Chapter 4: Unlocking Insights and Opportunities with NLP in Asset Management** by:
    1. **Applying NLP for sentiment analysis**: The app uses both **TextBlob** and **VADER** to perform sentiment analysis, which is one of the core concepts covered in the chapter.
    2. **Entity Recognition**: The chapter discusses how extracting entities like company names, stock symbols, and other financial terms from news can provide useful information for asset management strategies. This app uses **spaCy** for entity tagging.
    3. **Interactive Word Clouds**: The word clouds visually represent the most frequently identified entities based on sentiment, enhancing the interpretability of NLP outputs.
    4. **Real-time Insights**: By enabling users to input news text and instantly see sentiment analysis, entity tagging, and word clouds, the app aligns with Chapter 4's focus on using NLP for dynamic asset management strategies.

    Overall, this app demonstrates how NLP techniques can process financial news, extract sentiment, identify entities, and visualize results for asset management.
    """)
