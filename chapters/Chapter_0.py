import streamlit as st


def chapter0():

    st.subheader("AI and Big Data Applications in Investment Management")
    st.write("""
    Welcome to the application based on **'Handbook of Artificial Intelligence and Big Data Applications in Investments'** by Larry Cao, CFA. 
    This application contains dedicated modules that simulate the practical applications of AI and Big Data in various aspects of finance and investment, 
    aligned with the insights and case studies presented in each chapter of the handbook.
    """)

    # Reference to the original document
    st.write("### Reference")
    st.write("""
    This application is inspired by and references:
    - **[Cao, Larry. Handbook of Artificial Intelligence and Big Data Applications in Investments.](https://rpc.cfainstitute.org/en/research/foundation/2023/ai-and-big-data-in-investments-handbook)** CFA Institute Research Foundation, 2023.
    """)

    # Chapter Navigation and Overview
    st.write("### Chapter Applications")
    st.write("Each chapter's application can be accessed from the sidebar. These applications bring theoretical concepts to life with interactive simulations, filters, and visualizations.")

    chapters = {
        "Chapter 1: Machine Learning Applications in Investments": "Application of ML models in predicting returns and analyzing risk",
        "Chapter 2: Alternative Data and AI in Investment Research": "Using alternative data for generating investment insights",
        "Chapter 3: Data Science for Active and Long-Term Fundamental Investing": "Fundamental analysis powered by data science",
        "Chapter 4: NLP in Asset Management": "Natural language processing for sentiment analysis and client insights",
        "Chapter 5: Advances in NLU for Investment Management": "Exploring natural language understanding (NLU) in finance",
        "Chapter 6: Text-Based ESG Insights": "Extracting ESG insights using text-based analysis",
        "Chapter 7: Machine Learning for Trade Execution": "ML algorithms enhancing trade execution efficiency",
        "Chapter 8: Microstructure Data-Driven Execution": "Predicting spreads, volume, and volatility for trading",
        "Chapter 9: Intelligent Customer Service in Finance": "Implementing AI-driven customer service and quality inspection",
        "Chapter 10: Accelerated AI and Use Cases in Investment Management": "Accelerated computing in AI applications for portfolio management",
        "Chapter 11: Symbolic AI: A Case Study": "Applying rule-based symbolic AI for portfolio decisions"
    }

    for chapter, description in chapters.items():
        st.write(f"**{chapter}**")

    # Prompt to explore applications
    st.write("### Explore the Chapters")
    st.write("Click on each chapter in the sidebar to dive into an interactive application for that topic. Each section allows you to experiment with data and models related to the chapter's theme, helping you understand the impact of AI in investment management through hands-on simulations.")

    # Footer
