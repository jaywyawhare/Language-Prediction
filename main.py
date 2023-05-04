import nbformat
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import fasttext
import fasttext.util
from fasttext.FastText import _FastText
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components

model = load_model('./models/FT_MLP4.h5')


dic = {0 : 'Arabic', 1 : 'Danish', 2 : 'Dutch', 3 : 'English', 4 : 'French', 5 : 'German', 6 : 'Greek', 7 : 'Hindi', 8 : 'Italian', 9 : 'Kannada', 10 : 'Malayalam', 11 : 'Portugeese', 12 : 'Russian', 13 : 'Spanish', 14 : 'Sweedish', 15 : 'Tamil', 16 : 'Turkish'} 

ft = _FastText('./models/lid.bin')

def ftEmbedding(text):
    return ft.get_sentence_vector(text)


def preprocess_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("  ", " ")
    return text


st.sidebar.title("Select an option")
option = st.sidebar.radio("", ('Language Identification Tool', 't-SNE 2D Plot', 't-SNE 3D Plot'))

if option == 'Language Identification Tool':
    st.title("Language Identification Tool")

    text = st.text_area("Enter the text to be analysed")
    if st.button("Predict"):
        text = preprocess_text(text)
        if len(text) < 6:
            st.write("Please enter a long text")
        else:
            embedding = ftEmbedding(text)
            embedding = np.array(embedding)
            embedding = embedding.reshape(1, 16)
            prediction = model.predict(embedding)
            prediction = np.argmax(prediction)
            st.write("The language is of the text is: ", dic[prediction])

elif option == 't-SNE 3D Plot':
    st.title("t-SNE 3D Plot")
    HtmlFile = open("./html/plotly3D.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=600, width=600)

elif option == 't-SNE 2D Plot':
    st.title("t-SNE 2D Plot")
    HtmlFile = open("./html/plotly2D.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=600, width=600)

