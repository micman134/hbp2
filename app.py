# hbp_app.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import hashlib
import binascii
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="HBP Risk Prediction System", layout="wide")

# Hide default Streamlit UI
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .custom-footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Password hashing
# ----------------------------
def hash_password(password):
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password, provided_password):
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', 
                                provided_password.encode('utf-8'), 
                                salt.encode('ascii'), 
                                100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_password

# ----------------------------
# Firebase Initialization
# ----------------------------
def init_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(st.secrets["firebase"])
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase initialization error: {e}")
        return None

# ----------------------------
# Authentication Functions
# ----------------------------
def login_user(email, password):
    try:
        db = init_firebase()
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        if len(query) == 0:
            st.error("Invalid credentials")
            return False
        user_doc = query[0]
        user_data = user_doc.to_dict()
        if verify_password(user_data['password'], password):
            st.session_state.user = {
                'uid': user_doc.id,
                'email': email,
                'name': user_data.get('name')
            }
            return True
        else:
            st.error("Invalid credentials")
            return False
    except Exception as e:
        st.error(f"Login error: {e}")
        return False

def signup_user(email, password, name):
    try:
        db = init_firebase()
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        if len(query) > 0:
            st.error("Email already exists")
            return False
        hashed_password = hash_password(password)
        user_data = {
            'email': email,
            'name': name,
            'password': hashed_password,
            'created_at': datetime.datetime.now(datetime.timezone.utc)
        }
        doc_ref = users_ref.add(user_data)
        st.session_state.user = {
            'uid': doc_ref[1].id,
            'email': email,
            'name': name
        }
        return True
    except Exception as e:
        st.error(f"Signup error: {e}")
        return False

# ----------------------------
# Save Prediction
# ----------------------------
def save_prediction(input_data, prediction, proba):
    try:
        db = init_firebase()
        if not db:
            return False
        predictions_ref = db.collection('hbp_predictions')
        prediction_data = input_data.copy()
        prediction_data['prediction'] = 'High Risk' if prediction[0] == 1 else 'Low Risk'
        prediction_data['probability'] = float(proba[1])
        prediction_data['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
        if 'user' in st.session_state:
            prediction_data['user_id'] = st.session_state.user['uid']
        predictions_ref.add(prediction_data)
        return True
    except Exception as e:
        st.error(f"Error saving prediction: {e}")
        return False

# ----------------------------
# Load Predictions
# ----------------------------
def get_predictions():
    try:
        db = init_firebase()
        if not db:
            return None
        predictions_ref = db.collection('hbp_predictions')
        if 'user' in st.session_state:
            query = predictions_ref.where('user_id', '==', st.session_state.user['uid'])
        else:
            query = predictions_ref
        docs = query.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        predictions = []
        for doc in docs:
            d = doc.to_dict()
            d['id'] = doc.id
            predictions.append(d)
        return predictions
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return None

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model_scaler():
    model = joblib.load("hypertension_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model_scaler()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------------
# Initialize Session State
# ----------------------------
if 'user' not in st.session_state:
    st.session_state.user = None
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# ----------------------------
# Sidebar - Auth + Navigation
# ----------------------------
with st.sidebar:
    if st.session_state.user:
        st.success(f"Logged in as {st.session_state.user['email']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.submitted = False
            st.rerun()
    else:
        st.title("Authentication")
        auth_tab = st.tabs(["Login", "Sign Up"])
        with auth_tab[0]:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if login_user(email, password):
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
        with auth_tab[1]:
            with st.form("signup_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    if password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(password) < 8:
                        st.error("Password must be at least 8 characters")
                    elif signup_user(email, password, name):
                        st.success("Account created successfully!")
                        time.sleep(1)
                        st.rerun()

    if st.session_state.user:
        page = st.radio("Menu", ["Predict", "History", "Ontology", "About"])
    else:
        page = None

# ----------------------------
# Main Content
# ----------------------------
if not st.session_state.user:
    st.title("Welcome to HBP Risk Prediction System")
    st.write("Login to access the prediction tools.")
elif page == "About":
    st.title("About HBP Risk Prediction Tool")
    st.write("""
    - Machine learning model trained on 2,000+ records
    - Uses 13 key features
    - Predicts risk of hypertension
    """)
elif page == "Ontology":
    st.title("HBP Risk Factor Ontology")
    st.write("Visualizing risk factor relationships")
    st.image("https://raw.githubusercontent.com/your-username/your-repo/main/ontology2.png")
elif page == "History":
    st.title("Prediction History")
    preds = get_predictions()
    if preds:
        df = pd.DataFrame(preds)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.dataframe(df[['timestamp', 'prediction', 'probability']])
    else:
        st.info("No history found")
elif page == "Predict":
    st.title("HBP Risk Prediction Tool")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 45)
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            loh = st.number_input("Hemoglobin Level (g/dl)", 0.0, 20.0, 12.0)
            gpc = st.number_input("Inbreeding Coefficient", 0.0, 1.0, 0.0)
            pa = st.number_input("Physical Activity (CAL/4.18Kj)", 0, 2000)
            scid = st.number_input("Salt Intake (g/day)", 0.0, 20.0, 5.0)
        with col2:
            alcohol = st.number_input("Alcohol (ml/day)", 0, 500, 0)
            los = st.selectbox("Stress Level", ["Acute/Normal", "Episodic Acute", "Chronic"])
            ckd = st.selectbox("Chronic Kidney Disease", ["No", "Yes"])
            atd = st.selectbox("Thyroid Disorders", ["No", "Yes"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            pregnancy = st.selectbox("Pregnancy", ["No", "Yes"])
            smoking = st.selectbox("Smoking", ["No", "Yes"])
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = {
                'Age': age,
                'BMI': bmi,
                'LOH': loh,
                'GPC': gpc,
                'PA': pa,
                'salt_CID': scid,
                'alcohol_CPD': alcohol,
                'LOS': ["Acute/Normal", "Episodic Acute", "Chronic"].index(los) + 1,
                'CKD': 1 if ckd=="Yes" else 0,
                'ATD': 1 if atd=="Yes" else 0,
                'Sex': 1 if gender=="Female" else 0,
                'Pregnancy': 1 if pregnancy=="Yes" else 0,
                'Smoking': 1 if smoking=="Yes" else 0
            }
            X = pd.DataFrame([input_data])
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            proba = model.predict_proba(X_scaled)[0]
            st.success(f"Prediction: {'High Risk' if pred[0]==1 else 'Low Risk'}")
            st.info(f"Probability: {proba[1]:.2%}")
            save_prediction(input_data, pred, proba)
