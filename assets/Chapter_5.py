import streamlit as st
from transformers import pipeline
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Load spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Load BERT sentiment model from Hugging Face
sentiment_model = pipeline(
    'sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')


def chapter5():

    # App Title and Introduction
    st.header('Sentiment and Tone Analysis of Financial Documents')
    st.divider()
    st.markdown("""
    This app allows users to upload financial documents, perform sentiment analysis, analyze tone, and identify entities using pre-trained NLP models.
    It is aligned with **Chapter 5: Advances in Natural Language Understanding for Investment Management**, which emphasizes the use of advanced NLP models to extract insights from financial documents.
    """)

    # Upload Financial Document
    st.subheader('Upload a Financial Document')

    # add a note saying that for the purpose of the demo only the first 500 characters will be used
    st.markdown(
        ":red[For the purpose of this demo, only the first 500 characters of the document will be used for analysis.]")

    uploaded_file = st.file_uploader(
        'Upload a .txt or .pdf file', type=['txt', 'pdf'])

    if uploaded_file:
        # Read uploaded file content
        if uploaded_file.type == 'text/plain':
            document_text = uploaded_file.read().decode('utf-8')
        else:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            document_text = ""
            for page in pdf_reader.pages:
                document_text += page.extract_text()

        # Display file content
        st.subheader('Uploaded Document Content')
        st.write(document_text[:500] + '...')  # Display first 500 characters

        # Limit to 500 characters for processing
        document_text = document_text[:500]

        # Perform sentiment analysis using BERT
        st.subheader('Sentiment Analysis')
        sentiment_results = sentiment_model(document_text)
        overall_sentiment = sentiment_results[0]['label']

        # Display overall sentiment
        st.write(f"**Overall Sentiment**: {overall_sentiment}")

        # Tone Analysis using TextBlob
        st.subheader('Tone Analysis')
        blob = TextBlob(document_text)
        textblob_sentiment = blob.sentiment.polarity

        # Classify the tone based on TextBlob polarity
        if textblob_sentiment > 0.1:
            tone_label = 'Positive'
        elif textblob_sentiment < -0.1:
            tone_label = 'Negative'
        else:
            tone_label = 'Neutral'

        st.write(
            f"**Overall Tone**: {tone_label} (Polarity: {textblob_sentiment})")

        # Entity Recognition using spaCy
        st.subheader('Entity Recognition')
        doc = nlp(document_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Display entities
        if entities:
            for entity, label in entities:
                st.write(f"**Entity**: {entity}, **Type**: {label}")
        else:
            st.write("No entities detected.")

        # WordCloud for Entities
        st.subheader('Word Cloud of Entities')
        entity_text = ' '.join([ent[0] for ent in entities])

        if entity_text:
            entity_wordcloud = WordCloud(
                width=800, height=400, background_color='white').generate(entity_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(entity_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("No entities to generate a word cloud.")

    # Add explanation about relation to Chapter 5
    st.markdown("""
    ### Relation to Chapter 5
    This app reflects the learnings from **Chapter 5: Advances in Natural Language Understanding for Investment Management**:
    1. **Advanced NLP Models**: The app leverages advanced models like **BERT** for sentiment analysis and **spaCy** for entity recognition, which are discussed in the chapter.
    2. **Real-time Tone Analysis**: The tone analysis feature demonstrates how NLP can be used to extract tone from large volumes of financial documents, a key topic in Chapter 5.
    3. **Entity Recognition**: Identifying key entities in financial documents helps uncover insights about companies, stocks, and other financial terms, aligning with Chapter 5's focus on deeper understanding through NLP.

    Overall, this app serves as an interactive implementation of Chapter 5's concepts, showcasing how advanced NLP models can be used to process financial documents.
    """)
