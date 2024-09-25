# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved model files
scaler = joblib.load('scaler.pkl')
X_scaled = joblib.load('X_scaled.pkl')
y = joblib.load('y.pkl')
X_columns = joblib.load('X_columns.pkl')

# Function to recommend team and location based on cosine similarity
def recommend_team_location(new_patient_data, existing_data, target_labels, top_n=1):
    similarity_scores = cosine_similarity(new_patient_data, existing_data)
    top_n_indices = np.argsort(similarity_scores[0])[::-1][:top_n]
    recommended_team_location = target_labels.iloc[top_n_indices].mode()[0]
    return recommended_team_location

# Streamlit app interface
st.title("Patient Team & Location Recommendation System")

# Collect input from the user
referral_source = st.selectbox('Referral Source', ['Crisis', 'Primary Care'])
age_band = st.selectbox('Age Band', ['18-25', '26-35', '36-45', '46-55'])

# Convert user input into a format the model can use
new_patient = {
    'Referral source_Crisis': 1 if referral_source == 'Crisis' else 0,
    'Referral source_Primary Care': 1 if referral_source == 'Primary Care' else 0,
    'Age Band_18-25': 1 if age_band == '18-25' else 0,
    'Age Band_26-35': 1 if age_band == '26-35' else 0,
    'Age Band_36-45': 1 if age_band == '36-45' else 0,
    'Age Band_46-55': 1 if age_band == '46-55' else 0,
}

# Convert the new patient dictionary to a DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Ensure that all columns match the training data (X_columns)
# Missing columns will be set to 0. Reindex according to the scaler's expected feature names.
new_patient_df = new_patient_df.reindex(columns=X_columns, fill_value=0)

# Scale the new patient data
new_patient_scaled = scaler.transform(new_patient_df)

# Provide the recommendation when the button is clicked
if st.button('Recommend Team & Location'):
    recommended_team_location = recommend_team_location(new_patient_scaled, X_scaled, y)
    st.write(f'Recommended Team & Location: {recommended_team_location}')
