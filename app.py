import os 
import pandas as pd 
import numpy as np
import os 
import streamlit as st
from PIL import Image
from tempfile import NamedTemporaryFile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from inference import prediction
from inference import model 
from gradcam import GradCAM


# warnings streamlit 
st.set_option('deprecation.showfileUploaderEncoding', False)



# main menu
def menu():

    st.sidebar.header('Home')
    page = st.sidebar.radio("", ('GAVB Consulting',
                                 'IA aplicada a Saúde',
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

    if page == 'GAVB Consulting':
        gavb()

    if page == 'IA aplicada a Saúde':
        plataform()

    if page == 'Plataforma DL':
        dl_system()

    if page == 'Contato':
        contato()




def gavb():

    # Sobre GAVB 
    st.title('GAVB')
    st.write('Com foco em tecnologia e inovação, a GAVB é referência nacional em soluções que apoiam as organizações a inovarem com práticas data-driven business.'
             ' Com expertise para desenvolver projetos de alta complexidade, a GAVB é muito mais do que um fornecedor, é um trust advisor em tecnologia de seus clientes, com o objetivo de alavancar o uso de tecnologia dentro das empresas e extrair o máximo delas.')
    st.write('\n')
    st.write('\n')
    st.image(
        'https://www.vagastifloripa.com/wp-content/uploads/company_logos/2019/05/logo-novo-1.png',
        width=700,
        height=700,
        caption='GAVB Logo')




def dl_system():


    # texto da página 
    st.title('IA para Detecção de doenças pulmonares')
    st.write("\n ")
    st.write("O sistema de Inteligência Artificial desenvolvido tem a finalidade de identificar doenças pulmonares, auxiliando de maneira acurada o trabalho do médico.")
    st.write("Além de identificar se a imagem de um determinado raio-x há algum indício de uma doença, o sistema mostra onde ele olhou para tomar a decisão final sobre o raio-x, isso permite ao médico ter uma interpretabilidade."
            "de onde a IA está olhando, e possívelmente identificar possíveis ruídos que não foi identificado pelo médico.")
    st.write('\n')
    st.write('\n')

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
        #os.mkdir(f"{image}")
        #image = np.asarray(Image.open(temp_file))
        st.image(image, caption='Raio-x',
                    width=300,
                    height=250)


        # Botão de predição
        if st.button('Predição'):
            y_pred = prediction(image)
            if y_pred.any()==0:
                st.success("Predição: Covid")
            elif y_pred.any()==1:
                st.success("Predição: Normal")
            elif y_pred.any()==2: 
                st.success("Predição: Pneumonia")
    else: 
        pass 

    


    # parâmetros GradCAM 
    architecture = model
    last_conv = model.get_layer("conv5_block3_out")
    last_layers = ["avg_pool", "predictions"]
    image_size = (224, 224, 3)
    #img_path = image

    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.title("Intepretabilidade IA")
    st.write("\n")
    st.write("\n")
    st.write("\n")


    # GradCAM 
    if st.button("GradCAM"):
        img_path = uploaded_file.name
        grad = GradCAM(architecture,
                        last_conv, 
                        last_layers,
                        img_path,
                        image_size)
        image_grad = grad.gradcam_generate()
        st.image(image_grad, caption='Diagnóstico',
                    width=500,
                    height=450)


    


# IA aplicada a saúde 
def plataform():
    st.title('IA na Saúde')
    st.write('Com análises mais precisas, a IA fornece informações para que os diagnósticos sejam ainda mais eficientes. Sabemos que existem muitas doenças difíceis de serem diagnosticadas inicialmente, o que pode trazer prejuízos no tratamento do paciente.')
    st.write('\n')
    st.image('https://cdn.datafloq.com/cache/blog_pictures/2020/04/13/878x531/.conversational-ai-healthcare-2-key-use-casesjpg',
             width=600,
             height=300) # https://miro.medium.com/max/1200/1*m8yRo6dMb-KvHXe3qFt7wg.png
    st.write('\n')
    st.write('Com os avanços da inteligência artificial na área da saúde, o diagnóstico será mais preciso e rápido, o que salvará vidas, além de tornar os tratamentos mais diretos e seguros.'
    )
        



def contato():
    st.title('Contato')
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSh8_BbTxZSsHWdLsSVjvVGVjASl3WynpmbMg&usqp=CAU',
             width=100, height=100)
    st.write('\n')
    st.write('\n')
    st.write("Head of Computer Vision at GAVB: [Cássio Alcântara](https://www.linkedin.com/in/cassio-alcantara/?lipi=urn%3Ali%3Apage%3Acompanies_company_people_index%3BPicxypcpTSeRt3yPIrc%2BtA%3D%3D)")
    st.write('Esta Case foi desenvolvido pelo time de Visão Computacional da GAVB: [Luan Moreira](https://www.linkedin.com/in/luan-moreira-241ba1110/?lipi=urn%3Ali%3Apage%3Acompanies_company_people_index%3B8ubcT1F2QBWsv1etmIvEFQ%3D%3D), [Octavio Santana](https://www.linkedin.com/in/octavio-santana/?lipi=urn%3Ali%3Apage%3Acompanies_company_people_index%3BPicxypcpTSeRt3yPIrc%2BtA%3D%3D)  e [Felipe Oliveira](https://www.linkedin.com/in/felipe-oliveira-da-silva-18a573189/)')
    st.write('\n')
    
    st.write('\n')
    st.write('E-mail: @Gavb')
    





if __name__ == "__main__":
    menu()