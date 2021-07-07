# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 12:51:45 2021

@author: Nishant
"""

import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
classifier_pkl_in = open("./static/classifier.pkl", "rb")
classifier = pickle.load(classifier_pkl_in)

scaler_pkl_in = open("./static/standardscaler.pkl", "rb")
sc = pickle.load(scaler_pkl_in)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features = sc.transform(final_features)
    prediction = classifier.predict(final_features)
    return render_template('index.html', prediction_text='The Fish belongs to {} species '.format(prediction))


if __name__ == '__main__':
    app.run()
