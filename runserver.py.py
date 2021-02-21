

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

from skimage.transform import resize

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'covid.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH,compile = False)


#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

#Prediction
def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        a = preds[0]
        ind = np.argmax(a)
        index = ['COVID19','NORMAL','PNEUMONIA']
        print('Prediction:', index[ind])
        text = "prediction : "+index[ind]
        result = index[ind]
               # ImageNet Decode
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)


