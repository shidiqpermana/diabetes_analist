from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model dan preprocessor
model = load_model('model/ann_diabetes_model.h5')
preprocessor = joblib.load('model/preprocessor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = {
                'gender': request.form['gender'],
                'age': float(request.form['age']),
                'hypertension': int(request.form['hypertension']),
                'heart_disease': int(request.form['heart_disease']),
                'smoking_history': request.form['smoking_history'],
                'bmi': float(request.form['bmi']),
                'HbA1c_level': float(request.form['hba1c']),
                'blood_glucose_level': int(request.form['glucose'])
            }

            # Konversi ke DataFrame untuk preprocessing
            import pandas as pd
            input_df = pd.DataFrame([input_data])
            processed_input = preprocessor.transform(input_df)
            pred = model.predict(processed_input)[0][0]
            prediction = f"Kemungkinan Diabetes: {pred*100:.2f}%"

        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
