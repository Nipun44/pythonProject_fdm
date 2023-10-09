# from asgiref import simple_server
import pickle

import numpy as np
from flask import Flask, request, render_template, Response, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

model_lr=pickle.load(open('logisticRegression.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

from flask import request

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get form data and convert to integer values
        gender = int(request.form.get('gender'))
        age = int(request.form.get('age'))
        smoker = int(request.form.get('smoker'))
        cigsPerDay = int(request.form.get('cigsPerDay'))
        bloodPressureMedication = int(request.form.get('bloodPressureMedication'))
        prevalentStroke = int(request.form.get('prevalentStroke'))
        prevalentHyp = int(request.form.get('prevalentHyp'))
        diabetes = int(request.form.get('diabetes'))
        cholesterol = int(request.form.get('cholesterol'))
        sysBP = int(request.form.get('sysBP'))
        diaBP = int(request.form.get('diaBP'))
        BMI = int(request.form.get('BMI'))
        heartRate = int(request.form.get('heartRate'))
        glucose = int(request.form.get('glucose'))

        # Prepare the input data for prediction
        input_data = [[gender, age, smoker, cigsPerDay, bloodPressureMedication, prevalentStroke,
                       prevalentHyp, diabetes, cholesterol, sysBP, diaBP, BMI, heartRate, glucose]]
        print(input_data)

        # Make prediction using your machine learning model
        prediction = model_lr.predict(input_data)  # Assuming you have a loaded machine learning model named 'model'

        # You can now use the 'prediction' variable as the predicted output

        # Return the prediction to the client
        return str(prediction[0])  # This sends the prediction back as a string response

    else:
        # Handle GET request method if needed
        # ...
        pass

if __name__=='__main__':
    app.run()