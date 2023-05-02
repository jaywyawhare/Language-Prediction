import streamlit as st
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
import nbformat


import numpy as np

# model = load_model('./models/text.h5')


def preprocess_text(text):
    text = text.lower()
    return img


def app():
    st.title("Language Identification using Machine Learning")
    # st.header("A simple app to predict the language of a given text")
    # st.text("Enter the text to be analysed")
    text = st.text_area("Enter the text to be analysed")
    

    if st.button("Predict"):
        # text = preprocess_text(text)
        # prediction = model.predict(text)
        # st.write(prediction)
        prediction = 'Thoda ruko'
        st.write("The language of the given text is: ", prediction)


if __name__ == "__main__":
    app()