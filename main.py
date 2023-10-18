# from asgiref import simple_server
import pickle

import joblib
import numpy as np
from flask import Flask, request, render_template, Response, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

model_lr=pickle.load(open('logisticRegression.pkl', 'rb'))

dt = joblib.load('tree.pkl')
rfc = joblib.load('forest.pkl')

svm=pickle.load(open('SVM.pkl', 'rb'))





@app.route("/")
def index():
    return render_template('index.html')

from flask import request

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data from request

        data = request.form.to_dict()

        # Convert form data to the appropriate data types
        male = int(data['gender'])
        age = int(data['age'])
        currentSmoker = int(data['smoker'])
        if currentSmoker == 0:
            cigsPerDay = 0  # If not a current smoker, set cigarettes per day to 0
        else:
            cigsPerDay = float(data['cigsPerDay'])
        bloodPressureMedication = float(data['bloodPressureMedication'])
        prevalentStroke = int(data['prevalentStroke'])
        prevalentHyp = int(data['prevalentHyp'])
        diabetes = int(data['diabetes'])
        cholesterol = float(data['cholesterol'])
        sysBP = float(data['sysBP'])
        diaBP = float(data['diaBP'])
        BMI = float(data['BMI'])
        heartRate = float(data['heartRate'])
        glucose = float(data['glucose'])


        features = [male, age, currentSmoker, cigsPerDay, bloodPressureMedication,
                 prevalentStroke, prevalentHyp, diabetes, cholesterol,
                 sysBP, diaBP, BMI, heartRate, glucose]
        # print(cigsPerDay)
        # print(bloodPressureMedication)
        # print(male)


        # Make prediction using the loaded machine learning model
        prediction_lr = model_lr.predict([features])
        prediction_svm = svm.predict([features])
        prediction_dt = dt.predict([features])

        prediction_rfc = rfc.predict([features])

        # Return the prediction as a response
        # Modify prediction message based on the result

        predictions = [prediction_lr[0], prediction_svm[0], prediction_dt[0], prediction_rfc[0]]
        majority_prediction = max(set(predictions), key=predictions.count)

        # Generate prediction message based on the result
        if majority_prediction == 1:
            prediction_message = f"In the next 10 years, you have a chance to get a heart attack."
        else:
            prediction_message = f"In the next 10 years, you are not likely to get a heart attack."

        # Return the modified prediction message as a response
        return prediction_message
    else:
        return 'Invalid Request'

if __name__=='__main__':
    app.run()