import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="HBP Risk Prediction System", layout="wide")

# Hide Streamlit default UI
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('hypertension_model.pkl')  # replace with your model path
    scaler = joblib.load('scaler.pkl')            # replace with your scaler path
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Predict", "About"])

# About page
if page == "About":
    st.title("About This Tool")
    st.write("""
    **High Blood Pressure Risk Prediction Tool**  
    - Predicts hypertension risk based on demographic and health-related features  
    - Provides probability, risk factors, and care recommendations  
    - Uses a pre-trained machine learning model  
    """)

# Prediction page
else:
    st.title("High Blood Pressure Risk Prediction Tool")
    st.write("Fill in patient information and click 'Predict'.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            male = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age (years)", 0, 120, 45)
            currentSmoker = st.selectbox("Current Smoker?", ["No", "Yes"])
            cigsPerDay = st.number_input("Cigarettes Per Day", 0, 100, 0)
            BPMeds = st.selectbox("On Blood Pressure Medications?", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes?", ["No", "Yes"])

        with col2:
            totChol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
            sysBP = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
            diaBP = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
            BMI = st.number_input("Body Mass Index (kg/m²)", 10.0, 50.0, 25.0, step=0.1)
            heartRate = st.number_input("Heart Rate (bpm)", 40, 200, 70)
            glucose = st.number_input("Glucose (mg/dL)", 50, 300, 100)

        submitted = st.form_submit_button("Predict HBP Risk")

    if submitted:
        # Encode binary features
        input_data = {
            'male': 1 if male == "Male" else 0,
            'age': age,
            'currentSmoker': 1 if currentSmoker == "Yes" else 0,
            'cigsPerDay': cigsPerDay,
            'BPMeds': 1 if BPMeds == "Yes" else 0,
            'diabetes': 1 if diabetes == "Yes" else 0,
            'totChol': totChol,
            'sysBP': sysBP,
            'diaBP': diaBP,
            'BMI': BMI,
            'heartRate': heartRate,
            'glucose': glucose
        }

        features = ["male", "age", "currentSmoker", "cigsPerDay", "BPMeds", 
                    "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

        X = pd.DataFrame([[input_data[f] for f in features]], columns=features)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]

        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("High Risk of Hypertension")
            st.warning("Consider immediate clinical evaluation")
        else:
            st.success("Low Risk of Hypertension")
            st.info("Routine monitoring recommended")

        # Probability chart
        fig, ax = plt.subplots()
        ax.bar(['Low Risk', 'High Risk'], proba, color=['#2ecc71', '#e74c3c'], width=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Hypertension Risk Probability")
        for i, v in enumerate(proba):
            ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
        st.pyplot(fig)

        # Simple care suggestions
        st.subheader("Personalized Care Recommendations")
        if BMI >= 30:
            st.write("- Weight reduction program")
        if sysBP >= 140 or diaBP >= 90:
            st.write("- Blood pressure monitoring")
        if currentSmoker == "Yes":
            st.write("- Smoking cessation program")
        if diabetes == "Yes" or glucose > 126:
            st.write("- Monitor and manage blood sugar")
        if prediction[0] == 1:
            st.write("- Consult healthcare professional")
