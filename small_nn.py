import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing import image
import matplotlib.pyplot as plt
from scipy.misc import toimage
import pickle
from skimage import io
import os
from random import shuffle
from keras.models import model_from_json

class small_predict:
    def load_small_model():
        json_file = open('signs_small_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("signs_small_model.h5")
        print("Loaded model from disk")

        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print("model compile successfuly!")
        print(loaded_model.summary())
        return loaded_model

    def predict_class(img_path, model):
        classes = ['2.1', '2.4', '3.1', '3.24', '3.27', '4.1', '4.2', '5.19', '5.20', '8.22', '0']
        img = image.load_img(img_path, target_size=(32, 32))
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        print(prediction)
        prediction = np.argmax(prediction)
        print(classes[prediction])
        return classes[prediction]



