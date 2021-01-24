import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt 
import cv2 
from gradcam import GradCAM
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


# carregando rede 
model = tf.keras.models.load_model("C:\\Users\\Felipe Oliveira-GAVB\\Desktop\\Case_visao_computacional\\health\\saved_model_resnet\\resnet_model")



@st.cache(allow_output_mutation=True)
def prediction(image_path):

    # preprocessing image 
    #img = load_img(image_path, target_size=(224, 224))
    #x = img_to_array(image_path)
   # x = np.expand_dims(x, axis=0)
   # verificar type(image_path)
    x = np.resize(image_path, new_shape=[1, 224, 224, 3])
    x = preprocess_input(x)

    # prediction 
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred 

  #  if y_pred.any()==0:
  #    image = load_img(image_path)
  #    plt.figure(figsize=(8,5))
  #   plt.imshow(image)
  #    plt.title("Predição: Covid-19")
  #    plt.show()
  #  elif y_pred.any()==1:
  #    image = load_img(image_path)
  #    plt.figure(figsize=(8,5))
  #    plt.imshow(image)
  #    plt.title("Predição: Normal")
  #    plt.show()
  #  else: 
  #    image = load_img(image_path)
  #    plt.figure(figsize=(8,5))
  #    plt.imshow(image)
  #    plt.title("Predição: Pneumonia")
  #    plt.show()


