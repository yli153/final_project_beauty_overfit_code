import numpy as np
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt

def load_and_preprocess(path):
    img = Image.open(path)
    img = img.resize((400, 500),Image.ANTIALIAS)
    img = np.expand_dims(img,axis = 0)
    img = keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess(img):
    img = np.squeeze(img,0)
    #pixel mean from image net
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img,0,255).astype('uint8')
    return img

def save(best,path):
    img = Image.fromarray(best)
    img.save(path)