
import PyPDF2
import streamlit as st
from transformers import pipeline
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy.cli import download
import subprocess
from pathlib import Path


def chapter6():

    # Load the model
    import spacy
    from pathlib import Path
    import subprocess

    # Define a local directory to install the model
    model_path = Path("models/en_core_web_sm/en_core_web_sm-3.8.0")

    # Check if the model is already available locally
    if not model_path.exists():
        # Download and link the model to the local directory
        subprocess.run(["python3", "-m", "spacy", "download",
                       "en_core_web_sm", "--target", "models"])

    # Load the model from the local path
    nlp = spacy.load(str(model_path))

    sentiment_model = pipeline(
        'text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
    # App Title
    st.subheader(
        'Chapter 6: Extracting Text-Based ESG Insights: A Hands-On Guide')
    st.divider()
    st.markdown("""
    This application allows users to upload ESG documents and obtain dynamic ESG scoring based on NLP-driven thematic extraction. 
    It leverages the concepts discussed in **Chapter 6: Extracting Text-Based ESG Insights: A Hands-On Guide**, which highlights how NLP can be applied to process ESG documents for investment analysis.
    """)

    # File Uploader
    st.subheader('Upload Your ESG Document')

    st.markdown(
        ":red[For the purpose of this demo, only the first 500 characters of the document will be used for analysis.]")

    uploaded_file = st.file_uploader(
        "Upload a text or PDF file", type=["txt", "pdf"])

    if uploaded_file:
        # Read uploaded file
        if uploaded_file.type == 'text/plain':
            document_text = uploaded_file.read().decode('utf-8')
        else:

            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            document_text = ""
            for page in pdf_reader.pages:
                document_text += page.extract_text()

        # Display the first 500 characters of the document
        st.subheader("Uploaded Document Preview")
        st.write(document_text[:500] + "...")

        # Limit to 5000 characters for processing
        document_text = document_text[:500]

        # Thematic Extraction
        st.subheader("Thematic Extraction")
        themes = sentiment_model(document_text)
        st.write("Detected Themes:")
        for theme in themes:
            st.write(
                f"**Theme**: {theme['label']}, **Confidence**: {theme['score']:.2f}")

        # Generate ESG Score
        def generate_esg_score(themes):
            score = 0
            theme_weights = {
                'sustainability': 2.0,
                'environment': 2.0,
                'climate': 1.5,
                'carbon': 1.5,
                'social': 1.5,
                'diversity': 1.0,
                'inclusion': 1.0,
                'equality': 1.0,
                'governance': 2.0,
                'transparency': 1.5,
                'compliance': 1.5,
                'ethics': 1.0
            }

            for theme in themes:
                label = theme['label'].lower()
                for keyword, weight in theme_weights.items():
                    if keyword in label:
                        score += weight * theme['score']

            return score

        esg_score = generate_esg_score(themes)
        st.subheader(f"Overall ESG Score: {esg_score:.2f}")

        # Entity Recognition
        st.subheader("Entity Recognition")
        doc = nlp(document_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        if entities:
            with st.expander("View Entities"):
                st.write("Detected Entities:")
                for entity, label in entities:
                    st.write(f"**Entity**: {entity}, **Type**: {label}")
        else:
            st.write("No entities detected.")

        # WordCloud for Entities
        st.subheader('Word Cloud of Entities')
        entity_text = ' '.join([ent[0] for ent in entities])
        if entity_text:
            wordcloud = WordCloud(
                width=800, height=400, background_color='white').generate(entity_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("No entities to generate a word cloud.")

    else:
        st.write("Please upload a file to analyze.")

    # Add Relation to Chapter 6
    st.markdown("""
    ### Relation to Chapter 6: Extracting Text-Based ESG Insights
    The app aligns with the teachings of Chapter 6 by demonstrating:

    1. **Dynamic Analysis**: The chapter emphasizes the importance of dynamic and adaptive tools to analyze rapidly evolving ESG information. This app allows for real-time scoring based on the themes extracted from uploaded documents.
    2. **NLP for Thematic Extraction**: Chapter 6 explains the role of NLP in identifying ESG topics like sustainability, social responsibility, and governance at different levels. The thematic extraction in this app uses pre-trained models to highlight these themes from the text.
    3. **Scalable ESG Analysis**: By leveraging NLP, the app can process large volumes of unstructured text, mirroring the scalable approach described in the chapter.
    4. **Practical Implementation**: The chapter provides a hands-on guide for integrating NLP solutions with ESG problems, similar to the interactive nature of this application.
    """)
