import streamlit as st
from chapters.Chapter_0 import chapter0
from chapters.Chapter_1 import chapter1
from chapters.Chapter_2 import chapter2
from chapters.Chapter_3 import chapter3
from chapters.Chapter_4 import chapter4
from chapters.Chapter_5 import chapter5
from chapters.Chapter_6 import chapter6
from chapters.Chapter_7 import chapter7
from chapters.Chapter_8 import chapter8
from chapters.Chapter_9 import chapter9
from chapters.Chapter_10 import chapter10
from chapters.chapter_11 import chapter11

st.set_page_config(layout="wide")
st.write("# Welcome to AI and Big Data Applications in Investments Course!")
st.divider()


page_names_to_funcs = {
    "_": chapter0,
    "Chapter 1": chapter1,
    "Chapter 2": chapter2,
    "Chapter 3": chapter3,
    "Chapter 4": chapter4,
    "Chapter 5": chapter5,
    "Chatper 6": chapter6,
    "Chapter 7": chapter7,
    "Chapter 8": chapter8,
    "Chapter 9": chapter9,
    "Chapter 10": chapter10,
    "Chapter 11": chapter11
}
st.sidebar.image("assets/logo.jpg", use_column_width=True)
demo_name = st.sidebar.selectbox(
    "Choose a chapter", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

# add a footer with a copyright symbol and quantuniversity
st.markdown("""
---
©️ [QuantUniversity](https://www.quantuniversity.com), 2024
""")
