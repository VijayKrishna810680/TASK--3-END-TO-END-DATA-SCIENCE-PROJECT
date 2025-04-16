from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/diabetes_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([[data[key] for key in [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return jsonify({
        'prediction': int(prediction),
        'result': 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    })

@app.route('/predict_form', methods=['POST'])
def predict_form():
    form_data = request.form
    input_data = np.array([[float(form_data[key]) for key in [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    return f"<h2>Prediction: {result}</h2><br><a href='/'>Go Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
