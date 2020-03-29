import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/chord_suggestor_modelV2.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

chord_to_index = {'A': 0, 'A/C#': 1, 'A7': 2, 'A7/C#': 3, 'A7sus4': 4, 'Aadd9': 5, 'Am7': 6, 'Am7/F#': 7, 'Am7/G': 8, 'Am9': 9, 'Amadd9': 10, 'Asus2': 11, 'Asus4': 12, 'B': 13, 'B/Eb': 14, 'B7': 15, 'B7/Eb': 16, 'B7/F#': 17, 'Bb': 18, 'Bbm': 19, 'Bm': 20, 'Bm7': 21, 'Bm7/A': 22, 'Bm7b5': 23, 'Bsus4': 24, 'C': 25, 'C#/F': 26, 'C#m7': 27, 'C/Bb': 28, 'C/D': 29, 'C/E': 30, 'C/Eb': 31, 'C/G': 32, 'C7': 33, 'C7/E': 34, 'Cadd9': 35, 'Cm': 36, 'Cm/D': 37, 'Cm7/Bb': 38, 'Cm9/Eb': 39, 'Cmaj7': 40, 'Csus4': 41, 'D': 42, 'D#m': 43, 'D/B': 44, 'D/C': 45, 'D/E': 46, 'D/F#': 47, 'D7': 48, 'D7/F#': 49, 'D9': 50, 'Dm': 51, 'Dm7': 52, 'Dm7/A': 53, 'Dm7/Bb': 54, 'Dmaj7': 55, 'Dsus2': 56,
                  'Dsus4': 57, 'E': 58, 'E/D': 59, 'E/G#': 60, 'E7': 61, 'E7/G#': 62, 'Eb': 63, 'Em': 64, 'Em/C#': 65, 'Em/D': 66, 'Em11': 67, 'Em7': 68, 'Em9': 69, 'Em9/D': 70, 'Emaj7': 71, 'Esus4': 72, 'F': 73, 'F#': 74, 'F#/E': 75, 'F#m': 76, 'F#m/E': 77, 'F#m7': 78, 'F#m7b5': 79, 'F#m9': 80, 'F/A': 81, 'F/C': 82, 'F/G': 83, 'F9': 84, 'F9/A': 85, 'Fadd9': 86, 'Fm': 87, 'Fm(maj7)': 88, 'Fm/C': 89, 'Fm/G': 90, 'Fm/G#': 91, 'Fmaj7': 92, 'G': 93, 'G#': 94, 'G#m': 95, 'G#m/F': 96, 'G#maj7': 97, 'G#sus4': 98, 'G/B': 99, 'G/D': 100, 'G/F': 101, 'G11': 102, 'G7': 103, 'G7/B': 104, 'G7/D': 105, 'G7b9': 106, 'Gadd9': 107, 'Gm': 108, 'Gm7': 109, 'Gmaj7': 110, 'Gsus2': 111, 'Gsus4': 112}
index_to_chord = {0: 'A', 1: 'A/C#', 2: 'A7', 3: 'A7/C#', 4: 'A7sus4', 5: 'Aadd9', 6: 'Am7', 7: 'Am7/F#', 8: 'Am7/G', 9: 'Am9', 10: 'Amadd9', 11: 'Asus2', 12: 'Asus4', 13: 'B', 14: 'B/Eb', 15: 'B7', 16: 'B7/Eb', 17: 'B7/F#', 18: 'Bb', 19: 'Bbm', 20: 'Bm', 21: 'Bm7', 22: 'Bm7/A', 23: 'Bm7b5', 24: 'Bsus4', 25: 'C', 26: 'C#/F', 27: 'C#m7', 28: 'C/Bb', 29: 'C/D', 30: 'C/E', 31: 'C/Eb', 32: 'C/G', 33: 'C7', 34: 'C7/E', 35: 'Cadd9', 36: 'Cm', 37: 'Cm/D', 38: 'Cm7/Bb', 39: 'Cm9/Eb', 40: 'Cmaj7', 41: 'Csus4', 42: 'D', 43: 'D#m', 44: 'D/B', 45: 'D/C', 46: 'D/E', 47: 'D/F#', 48: 'D7', 49: 'D7/F#', 50: 'D9', 51: 'Dm', 52: 'Dm7', 53: 'Dm7/A', 54: 'Dm7/Bb', 55: 'Dmaj7', 56: 'Dsus2',
                  57: 'Dsus4', 58: 'E', 59: 'E/D', 60: 'E/G#', 61: 'E7', 62: 'E7/G#', 63: 'Eb', 64: 'Em', 65: 'Em/C#', 66: 'Em/D', 67: 'Em11', 68: 'Em7', 69: 'Em9', 70: 'Em9/D', 71: 'Emaj7', 72: 'Esus4', 73: 'F', 74: 'F#', 75: 'F#/E', 76: 'F#m', 77: 'F#m/E', 78: 'F#m7', 79: 'F#m7b5', 80: 'F#m9', 81: 'F/A', 82: 'F/C', 83: 'F/G', 84: 'F9', 85: 'F9/A', 86: 'Fadd9', 87: 'Fm', 88: 'Fm(maj7)', 89: 'Fm/C', 90: 'Fm/G', 91: 'Fm/G#', 92: 'Fmaj7', 93: 'G', 94: 'G#', 95: 'G#m', 96: 'G#m/F', 97: 'G#maj7', 98: 'G#sus4', 99: 'G/B', 100: 'G/D', 101: 'G/F', 102: 'G11', 103: 'G7', 104: 'G7/B', 105: 'G7/D', 106: 'G7b9', 107: 'Gadd9', 108: 'Gm', 109: 'Gm7', 110: 'Gmaj7', 111: 'Gsus2', 112: 'Gsus4'}

chord_space = len(chord_to_index)+1


def chords_to_one_hot(cs):
    one_hot = np.zeros((len(cs), chord_space))
    for i in range(len(cs)):
        c = cs[i]
        index = chord_to_index[c]
        # ith chord, index
        one_hot[i][index] = 1
    return one_hot


def model_predict(chord_sequence, model):
    Tx = len(chord_sequence)
    x_pred = np.zeros((1, Tx, chord_space))
    print("chord_space" + str(chord_space))
    x_pred[0] = chords_to_one_hot(chord_sequence)

    prediction = model.predict(x_pred)
    prediction = prediction[0][-1]

    # get the top k most probable chords
    top_k = 5
    max_prob_indices = np.argsort(prediction)[::-1][:top_k]
    top_chords = []
    for index in max_prob_indices:
        top_chords.append(index_to_chord[index])

    return top_chords


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        posted_input = request.form['text']
        sequence = posted_input.split()

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(sequence, model)

        # Process your result for human
  # pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
   #     pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(preds)               # Convert to string
        print(result)

        # Serialize the result, you can add additional fields
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
