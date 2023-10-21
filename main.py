# from asgiref import simple_server
import pickle
from statistics import mode

import joblib
import numpy as np
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__)

model_lr=pickle.load(open('logisticRegression_best.pkl', 'rb'))

dt = joblib.load('tree_best.pkl')
rfc = joblib.load('forest_best.pkl')

svm=pickle.load(open('SVM_best.pkl', 'rb'))





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
        prediction_lr = model_lr.predict([features])[0]
        prediction_svm = svm.predict([features])[0]
        prediction_dt = dt.predict([features])[0]

        prediction_rfc = rfc.predict([features])[0]

        # Collect predictions from all models
        predictions = [prediction_lr, prediction_svm, prediction_dt, prediction_rfc]

        # Calculate majority prediction using mode
        majority_prediction = mode(predictions)




        # Convert int64 objects to regular Python integers
        prediction_lr = int(prediction_lr)
        prediction_svm = int(prediction_svm)
        prediction_dt = int(prediction_dt)
        prediction_rfc = int(prediction_rfc)
        majority_prediction = int(majority_prediction)

        sendings = {
            "prediction_lr": prediction_lr,
            "prediction_svm": prediction_svm,
            "prediction_dt": prediction_dt,
            "prediction_rfc": prediction_rfc,
            "majority_prediction": majority_prediction

        }



        # Return the predictions as a response
        return jsonify(sendings)
    else:
        return 'Invalid Request'

if __name__=='__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)