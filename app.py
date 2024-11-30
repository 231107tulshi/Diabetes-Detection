from flask import Flask, render_template, request
from diabetes_model import predict_diabetes

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        result = predict_diabetes(input_data)
        return render_template('index.html', result=result)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
