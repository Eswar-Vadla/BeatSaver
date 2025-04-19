import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model


# Symbolic Rules

def symbolic_explanation(features, label):
    explanation = []
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features

    if label == 1:  # Heart disease is predicted
        explanation.append("Signs indicate a high risk or presence of heart disease based on your health profile.")

        # Age
        if age >= 65:
            explanation.append(f"At {age} years, the risk of heart disease increases significantly due to aging vessels and reduced cardiac efficiency.")
        elif age >= 50:
            explanation.append(f"At {age} years, regular monitoring is essential as age is a non-modifiable risk factor.")

        # Sex
        if sex == 1:
            explanation.append("Male individuals are generally more prone to heart disease, especially after age 45.")

        # Chest Pain (cp)
        cp_types = {
            0: "Typical Angina (most concerning)",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic (can be dangerous if silent symptoms)"
        }
        explanation.append(f"Chest pain type indicates {cp_types.get(cp, 'Unknown')}.")
        if cp == 0 or cp == 3:
            explanation.append("This pain pattern may reflect ischemic heart conditions.")

        # Resting Blood Pressure (trestbps)
        if trestbps >= 150:
            explanation.append(f"Severely elevated blood pressure ({trestbps} mmHg) may strain the heart and damage arteries.")
        elif trestbps >= 130:
            explanation.append(f"Blood pressure ({trestbps} mmHg) is in the hypertensive range.")

        # Cholesterol
        if chol >= 240:
            explanation.append(f"Cholesterol level of {chol} mg/dL is considered dangerously high, increasing risk of atherosclerosis.")
        elif chol >= 200:
            explanation.append(f"High cholesterol ({chol} mg/dL) can contribute to arterial plaque build-up.")

        # Fasting Blood Sugar
        if fbs == 1:
            explanation.append("High fasting blood sugar (>120 mg/dL) is a sign of diabetes, which correlates with heart disease.")

        # Resting ECG
        if restecg == 1:
            explanation.append("ECG indicates ST-T abnormality, suggesting possible past myocardial ischemia or infarction.")
        elif restecg == 2:
            explanation.append("Left ventricular hypertrophy seen on ECG may reflect long-standing hypertension or structural disease.")

        # Max Heart Rate (thalach)
        if thalach < 100:
            explanation.append(f"Low peak heart rate ({thalach} bpm) suggests reduced cardiovascular fitness.")
        elif thalach > 180:
            explanation.append(f"Extremely high peak heart rate ({thalach} bpm) may indicate abnormal cardiac stress response.")

        # Exercise-induced angina
        if exang == 1:
            explanation.append("Chest pain during exertion reflects poor blood flow to heart under stress.")

        # ST depression (oldpeak)
        if oldpeak > 2.5:
            explanation.append(f"ST depression of {oldpeak} mm indicates severe myocardial ischemia.")
        elif oldpeak > 1.0:
            explanation.append(f"Moderate ST depression ({oldpeak}) may reflect coronary artery blockage.")

        # Slope of ST segment
        if slope == 0:
            explanation.append("Flat ST slope is correlated with decreased ventricular performance.")
        elif slope == 1:
            explanation.append("Downward sloping ST is often seen in myocardial ischemia.")

        # Number of major vessels colored (ca)
        if ca >= 2:
            explanation.append(f"{ca} major vessels show calcification—this is a strong indicator of significant coronary artery disease.")

        # Thalassemia status
        if thal == 1:
            explanation.append("Fixed defect in thallium scan suggests previous infarction.")
        elif thal == 2:
            explanation.append("Normal thalassemia scan — not a concern by itself.")
        elif thal == 3:
            explanation.append("Reversible defect on scan indicates ischemia, potentially reversible with treatment.")

    else:  # No Heart Disease
        explanation.append("No signs of heart disease. Keep up the healthy habits!")

        # Age
        if age < 35:
            explanation.append(f"At {age} years, the risk of heart disease is generally low, and the heart remains highly efficient.")
        elif age < 50:
            explanation.append(f"At {age} years, maintaining a healthy lifestyle can prevent future heart disease risks.")

        # Resting Blood Pressure (trestbps)
        if trestbps < 120:
            explanation.append(f"Your resting blood pressure ({trestbps} mmHg) is in a healthy range, which is ideal for heart health.")

        # Cholesterol
        if chol < 180:
            explanation.append(f"Cholesterol level of {chol} mg/dL is excellent and protective against arterial disease.")
        elif chol < 200:
            explanation.append(f"Cholesterol level of {chol} mg/dL is within the normal range, offering heart protection.")

        # Max Heart Rate (thalach)
        if thalach >= 140:
            explanation.append(f"High peak heart rate ({thalach} bpm) during activity suggests good cardiovascular fitness and heart health.")

        # Exercise-induced angina
        if exang == 0:
            explanation.append("No angina during exercise indicates good blood supply to the heart and healthy cardiac function.")

        # ST depression (oldpeak)
        if oldpeak < 1.0:
            explanation.append(f"Low ST depression ({oldpeak}) suggests good heart perfusion and low risk for heart disease.")

        # Slope of ST segment
        if slope == 2:
            explanation.append("Upward ST slope is a positive indicator of healthy heart function and normal cardiac response.")

        # Number of major vessels colored (ca)
        if ca == 0:
            explanation.append("No calcified major vessels—an excellent sign of cardiovascular health and good circulation.")

        # Thalassemia status
        if thal == 2:
            explanation.append("Normal thallium scan, indicating healthy cardiac tissue and no evidence of ischemia.")

        # **Additional positive indicators:**

        # Healthy Lifestyle Habits
        # No smoking
        if "smoking" not in features:
            explanation.append("No smoking is a critical factor in reducing heart disease risk. Well done for maintaining this healthy habit.")
        # No excessive alcohol consumption
        if "alcohol" not in features or features.get("alcohol", 0) == 0:
            explanation.append("Moderate or no alcohol consumption supports good heart health and reduces the risk of heart disease.")
        # Physical activity
        if "exercise" in features and features["exercise"] >= 150:
            explanation.append(f"Regular physical activity ({features['exercise']} minutes per week) is associated with lower heart disease risk.")

        # Healthy Diet
        if "diet" in features and features["diet"] == "healthy":
            explanation.append("Eating a heart-healthy diet rich in fruits, vegetables, and whole grains supports long-term cardiovascular health.")

        # Family history
        if "family_history" in features and features["family_history"] == 0:
            explanation.append("No family history of heart disease is a positive indicator for a lower risk.")

        # Healthy weight (BMI check or body composition)
        if "bmi" in features and features["bmi"] < 25:
            explanation.append(f"Your BMI of {features['bmi']} indicates a healthy weight, which reduces the risk of heart disease.")

        explanation.append("Keep maintaining a heart-healthy lifestyle: exercise, a balanced diet, stress control, and regular check-ups.")

    return explanation if explanation else ["No significant indicators detected."]


# Symbolic Rules End

flask_app = Flask(__name__)
model = load_model("heart_disease_fixed.keras")

@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input features from the form
        input_values = [float(x) for x in request.form.values()]
        input_array = np.array([input_values])
        scaler = joblib.load('scaler.pkl')
        scaled_input_array = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input_array)
        label = int(prediction[0][0] > 0.5)

        # Interpret prediction
        result = "Heart Disease Detected" if label == 1 else "No Heart Disease Detected"
        explanation = symbolic_explanation(input_values, label)

        # Render the HTML with prediction and explanation
        return render_template("index.html", prediction_text=result, explanation_list=explanation)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e), explanation_list=[])


if __name__ == "__main__":
    flask_app.run(debug=True)
