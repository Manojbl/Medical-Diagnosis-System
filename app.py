import streamlit as st
import pandas as pd
import pickle

# Load all trained models
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
thyroid_model = pickle.load(open("models/thyroid_model.pkl", "rb"))
parkinsons_model = pickle.load(open("models/parkinsons_model.pkl", "rb"))
lung_model = pickle.load(open("models/lung_cancer_model.pkl", "rb"))

# App title
st.title("🩺 Medical Diagnosis System")
st.write("Predict Diabetes, Thyroid, Parkinson's, and Lung Cancer.")

# Sidebar to select disease
disease = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Thyroid", "Parkinson's", "Lung Cancer"]
)

# Function to make prediction
def predict(model, input_df):
    prediction = model.predict(input_df)
    return "Positive (1)" if prediction[0] == 1 else "Negative (0)"

# ---------------- Diabetes ----------------
if disease == "Diabetes":
    st.header("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    if st.button("Predict Diabetes"):
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree, age]],
                                  columns=["Pregnancies","Glucose","BloodPressure",
                                           "SkinThickness","Insulin","BMI",
                                           "DiabetesPedigreeFunction","Age"])
        result = predict(diabetes_model, input_data)
        st.success(f"Diabetes Prediction: {result}")

# ---------------- Thyroid ----------------
elif disease == "Thyroid":
    st.header("Thyroid Prediction")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["M", "F"])
    tsh = st.number_input("TSH Level", value=1.0)
    t3 = st.number_input("T3 Level", value=1.0)
    tt4 = st.number_input("TT4 Level", value=100.0)
    fti = st.number_input("FTI Level", value=100.0)

    if st.button("Predict Thyroid"):
        sex_val = 1 if sex == "M" else 0
        input_data = pd.DataFrame([[age, sex_val, tsh, t3, tt4, fti]],
                                  columns=["AGE","SEX","TSH","T3","TT4","FTI"])
        result = predict(thyroid_model, input_data)
        st.success(f"Thyroid Prediction: {result}")

# ---------------- Parkinson's ----------------
elif disease == "Parkinson's":
    st.header("Parkinson's Prediction")
    # For simplicity, we use 22 numeric features from the Parkinson dataset
    features = []
    columns = parkinsons_model.feature_names_in_
    for col in columns:
        val = st.number_input(f"{col}", value=0.0)
        features.append(val)
    if st.button("Predict Parkinson's"):
        input_data = pd.DataFrame([features], columns=columns)
        result = predict(parkinsons_model, input_data)
        st.success(f"Parkinson's Prediction: {result}")

# ---------------- Lung Cancer ----------------
elif disease == "Lung Cancer":
    st.header("Lung Cancer Prediction")
    # Example inputs for Lung Cancer (categorical yes/no)
    gender = st.selectbox("Gender", ["M", "F"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety", ["Yes", "No"])
    peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
    chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    allergy = st.selectbox("Allergy", ["Yes", "No"])
    wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consuming", ["Yes", "No"])
    coughing = st.selectbox("Coughing", ["Yes", "No"])
    shortness = st.selectbox("Shortness of Breath", ["Yes", "No"])
    swallowing = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])
    age = st.number_input("Age", min_value=0, max_value=120, value=50)

    if st.button("Predict Lung Cancer"):
        input_data = pd.DataFrame([[1 if gender=="M" else 0,
                                    1 if smoking=="Yes" else 0,
                                    1 if yellow_fingers=="Yes" else 0,
                                    1 if anxiety=="Yes" else 0,
                                    1 if peer_pressure=="Yes" else 0,
                                    1 if chronic_disease=="Yes" else 0,
                                    1 if fatigue=="Yes" else 0,
                                    1 if allergy=="Yes" else 0,
                                    1 if wheezing=="Yes" else 0,
                                    1 if alcohol=="Yes" else 0,
                                    1 if coughing=="Yes" else 0,
                                    1 if shortness=="Yes" else 0,
                                    1 if swallowing=="Yes" else 0,
                                    1 if chest_pain=="Yes" else 0,
                                    age]],
                                  columns=lung_model.feature_names_in_)
        result = predict(lung_model, input_data)
        st.success(f"Lung Cancer Prediction: {result}")
