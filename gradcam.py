import cv2
import numpy as np 
from io import StringIO 
import io
import base64
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img




class GradCAM(object):

  def __init__(self, architecture, last_conv, last_layers, img_path, img_size):
    self.architecture = architecture
    self.last_conv = last_conv
    self.last_layers = last_layers
    self.img_path = img_path
    self.img_size = img_size



  def gradcam_generate(self):

    # image 
    image = np.array(load_img(self.img_path, target_size=self.img_size))

    # last Conv2D layer 
    last_conv_layer_model = Model(self.architecture.inputs, self.last_conv.output)

    # create model with outputs 
    classifier_input = Input(shape=self.last_conv.output.shape[1:])
    x = classifier_input
    for layer_name in [self.last_layers[0], self.last_layers[1]]:
        x = self.architecture.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)


    # gradient derivative 
    with tf.GradientTape() as tape:
        inputs = image[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]


    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))


    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    ooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]


    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (224, 224))

    # visualize GradCAM
    plt.imshow(image)
    plt.imshow(gradcam, alpha=0.5, cmap='PuRd')
    #plt.savefig("gram_cam")

    # image to Bytes 
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_gram = Image.open(buf)


    return img_gram
    