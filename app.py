import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Custom page config
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

# Inject custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stSlider .st-cb {
        color: #333;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('ANN_model.keras')

# Load the encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('OHE_encoder.pkl', 'rb') as file:
    OHE_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title
st.title('📊 Customer Churn Prediction')

st.markdown("### Fill out the customer details below:")

# Split UI into two columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', OHE_encoder.categories_[0])
    gender = st.radio('👤 Gender', label_encoder.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=600)

with col2:
    balance = st.number_input('💰 Balance', min_value=0.0, value=0.0)
    estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0, value=50000.0)
    tenure = st.slider('📅 Tenure (Years with Bank)', 0, 10, 3)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)

with st.expander("🔧 Additional Settings"):
    has_cr_card = st.radio('Credit Card?', ['No', 'Yes'])
    is_active_member = st.radio('Active Member?', ['No', 'Yes'])

    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

# Prepare input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = OHE_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OHE_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)
churn_proba = prediction[0][0]

# Display prediction result
st.subheader("🔍 Prediction Result")

st.progress(float(churn_proba))
st.metric(label="Churn Probability", value=f"{churn_proba*100:.2f} %")

if churn_proba > 0.5:
    st.error('⚠️ The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')