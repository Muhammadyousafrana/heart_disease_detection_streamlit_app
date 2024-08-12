import streamlit as st
import pandas as pd
import joblib

# Load the models
rf_model = joblib.load('./Random_Forest.pkl')
dt_model = joblib.load('Decision_Tree.pkl')
xgb_model = joblib.load('./XGBoost.pkl')

# Create a model selection dropdown
model_choice = st.selectbox(
    "Select the model you want to use for prediction:",
    ("Random Forest", "Decision Tree", "XGBoost")
)

# Based on the user's choice, set the model to be used
if model_choice == "Random Forest":
    model = rf_model
elif model_choice == "Decision Tree":
    model = dt_model
else:
    model = xgb_model

# Define the input fields for each feature
st.write("Please enter the following information:")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0, 1, 2, 3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0, 1, 2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment (0, 1, 2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thal (0 = normal; 1 = fixed defect; 2 = reversable defect)", [0, 1, 2])

# Create a button for prediction
if st.button("Predict"):
    # Prepare the data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Make the prediction
    prediction = model.predict(input_data)[0]

    # Display the result
    if prediction == 1:
        st.write("The model predicts that you **have** heart disease.")
    else:
        st.write("The model predicts that you **do not have** heart disease.")
