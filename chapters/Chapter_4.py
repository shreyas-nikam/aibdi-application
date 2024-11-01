import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import re

def chapter4():
    # Define a local directory to install the model
    model_path = Path("models/en_core_web_sm/en_core_web_sm-3.8.0")

    # Check if the model is already available locally
    if not model_path.exists():
        # Download and link the model to the local directory
        subprocess.run(["python3", "-m", "spacy", "download",
                       "en_core_web_sm", "--target", "models"])

    # Load the model from the local path
    nlp = spacy.load(str(model_path))

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
        vader_sentiment = vader_analyzer.polarity_scores(user_input)['compound']

        # Entity Recognition and Sentiment Context Analysis
        doc = nlp(user_input)
        positive_entities = []
        negative_entities = []
        neutral_entities = []
        positive_words = []
        negative_words = []
        neutral_words = []

        # For each sentence, analyze the sentiment and extract contributing words
        for sentence in doc.sents:
            sentiment = vader_analyzer.polarity_scores(sentence.text)['compound']
            
            # Extract entities in this sentence
            for ent in sentence.ents:
                if sentiment > 0.1:
                    positive_entities.append(ent.text)
                elif sentiment < -0.1:
                    negative_entities.append(ent.text)
                else:
                    neutral_entities.append(ent.text)

            # Tokenize the sentence to add contributing words to word cloud lists
            words = [token.text for token in sentence if not token.is_stop and not token.is_punct]
            if sentiment > 0.1:
                positive_words.extend(words)
            elif sentiment < -0.1:
                negative_words.extend(words)
            else:
                neutral_words.extend(words)

        # Display Results
        st.subheader('Analysis Results')

        # Display sentiment scores
        st.markdown("### Sentiment Analysis")
        st.write(f"**TextBlob Sentiment Polarity**: {textblob_sentiment}")
        st.write(f"**VADER Sentiment Compound Score**: {vader_sentiment}")

        # Sentiment Classification
        if vader_sentiment > 0.1:
            sentiment_label = ':green[Positive]'
        elif vader_sentiment < -0.1:
            sentiment_label = ':red[Negative]'
        else:
            sentiment_label = ':orange[Neutral]'

        st.markdown(f"**Overall Sentiment**: {sentiment_label}")

        # Display entity recognition with scrollable container
        st.markdown("### Entity Recognition")
        if doc.ents:
            with st.expander("Detected Entities (Click to expand)"):
                for entity in doc.ents:
                    st.write(f"**Entity**: {entity.text}, **Type**: {entity.label_}")
        else:
            st.write("No entities detected.")

        # WordCloud Generation
        st.subheader('Word Clouds for Sentiment Analysis')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Positive Words and Entities")
            if positive_words or positive_entities:
                positive_content = positive_words + positive_entities
                positive_wordcloud = WordCloud(
                    width=400, height=300, background_color='white').generate(' '.join(positive_content))
                plt.figure(figsize=(5, 3))
                plt.imshow(positive_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No positive words or entities detected.")

        with col2:
            st.markdown("#### Neutral Words and Entities")
            if neutral_words or neutral_entities:
                neutral_content = neutral_words + neutral_entities
                neutral_wordcloud = WordCloud(
                    width=400, height=300, background_color='white').generate(' '.join(neutral_content))
                plt.figure(figsize=(5, 3))
                plt.imshow(neutral_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No neutral words or entities detected.")

        with col3:
            st.markdown("#### Negative Words and Entities")
            if negative_words or negative_entities:
                negative_content = negative_words + negative_entities
                negative_wordcloud = WordCloud(
                    width=400, height=300, background_color='white').generate(' '.join(negative_content))
                plt.figure(figsize=(5, 3))
                plt.imshow(negative_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No negative words or entities detected.")

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

