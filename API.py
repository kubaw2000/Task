### 6. Create a very simple REST API that will serve your models

#import necessary packages
import numpy as np
from flask import Flask, jsonify, request
import joblib

#create an app
app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify():

    # choice of model by user
    choice = request.json['choice']

    # loading of chosen model
    if choice == 'heuristic':
        model = joblib.load('heuristic.pkl')
    elif choice == 'knn':
        model = joblib.load('knn.pkl')
    elif choice == 'tree':
        model = joblib.load('tree.pkl')
    elif choice == 'neural_network':
        model = joblib.load('neural_network.pkl')
    else:
        return jsonify({'error': "Model not found. Available choices: 'heuristic', 'knn', 'tree',  'neural_network'"})

    # get input from the user input
    input = request.json['input_features']

    # predict
    prediction = model.predict(input)
    if choice == 'neural_network':
        prediction=np.argmax(prediction,axis=1)
    return jsonify({'prediction': prediction.tolist()})