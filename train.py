import cv2
import os 
import numpy as np 
from io import StringIO 
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img



path = "C:\\Users\\Felipe Oliveira-GAVB\Desktop\\case_gavb\health\\COVID-19 Radiography Database\\COVID.png"

image = load_img(path, target_size=(224,224,3))
image = array_to_img(image)
plt.imshow(image)
plt.savefig("NEW_COVID")

if image: 
    print("Imagem aqui")
else: 
    print("Sem imagem")