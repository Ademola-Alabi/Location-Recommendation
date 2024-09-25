# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the CSV files (assuming you've loaded the same data before)
calendar_data = pd.read_csv('calendar_data.csv')
contacts_data = pd.read_csv('contacts_data.csv')
episode_data = pd.read_csv('episode_data.csv')
patient_data = pd.read_csv('patient_data.csv')
service_directory_data = pd.read_csv('service_directory_data.csv')

# Step 1: Merging the DataFrames
merged_data = pd.merge(episode_data, contacts_data, on='Episode ID', how='left')
merged_data = pd.merge(merged_data, patient_data, on='Patient ID', how='left')
merged_data = pd.merge(merged_data, service_directory_data, left_on='Service ID Code', right_on='Service ID Code', how='left')

# Step 2: Create 'Referral to Intervention (Days)' column
merged_data['Referral to Intervention (Days)'] = pd.to_datetime(merged_data['First Intervention Date']) - pd.to_datetime(merged_data['Date'])
merged_data['Referral to Intervention (Days)'] = merged_data['Referral to Intervention (Days)'].dt.days

# Step 3: Combine 'Team Name' and 'Location' to create a single label (for recommendations)
merged_data['Team_Location'] = merged_data['Team Name'] + ' - ' + merged_data['Location']

# Step 4: Selecting relevant features for recommendation
recommendation_data = merged_data[['Referral source', 'Age Band', 'Team_Location']].dropna()

# Step 5: Encoding categorical variables
recommendation_data_encoded = pd.get_dummies(recommendation_data, columns=['Referral source', 'Age Band'])

# Step 6: Extract the feature vectors and the target label (Team_Location)
X = recommendation_data_encoded.drop(columns='Team_Location')
y = recommendation_data_encoded['Team_Location']

# Step 7: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Save the Scaler, Data, and Columns
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_scaled, 'X_scaled.pkl')
joblib.dump(y, 'y.pkl')
joblib.dump(X.columns, 'X_columns.pkl')  # Save the feature names for reindexing in the Streamlit app