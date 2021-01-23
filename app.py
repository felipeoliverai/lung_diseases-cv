import os 
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing.image import load_img


# warnings streamlit 
st.set_option('deprecation.showfileUploaderEncoding', False)






# main menu
def menu():

    st.sidebar.header('Home')
    page = st.sidebar.radio("", ('GAVB Consulting',
                                 'Case Saúde',
                                 'Plataforma DL',
                                 'Contato'))
    # hide the menu
    hide_streamlit_style = """
                <style>
                # MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    #if page == 'GAVB Consulting':
        #scicrop()

    #if page == 'Case Saúde':
        #plataform()

    if page == 'Plataforma DL':
        dl_system()

    #if page == 'Contato':
       # contato()




def dl_system():
    # upload da imagem 
    uploaded_file = st.file_uploader("Escolha uma imagem...", type="png")
    temp_file = NamedTemporaryFile(delete=False)
    st.write("\n")
    st.write("\n")
    st.write("\n")

    # carregamento da image 
    if uploaded_file is not None: 
        temp_file.write(uploaded_file.getvalue())
        image = Image.open(temp_file)
        st.image(image, caption='Raio-x',
                width=300,
                height=250)


    





if __name__ == "__main__":
    menu()