import pandas as pd
import os

# Define input and output paths
downloads_path = os.path.expanduser('~/Downloads')
desktop_path = os.path.expanduser('~/Desktop')

# Dataset filenames
file1 = 'predictive_healthcare_dataset_large-1.csv'
file2 = 'urban_bengaluru_dataset_large-1.csv'

# Load datasets
df_health = pd.read_csv(os.path.join(downloads_path, file1))
df_urban = pd.read_csv(os.path.join(downloads_path, file2))

# --- Cleaning Predictive Healthcare Dataset ---

# 1. Drop duplicates
df_health.drop_duplicates(inplace=True)

# 2. Handle missing values: drop rows with essential missing data or fill if reasonable
df_health.dropna(subset=['Patient_ID'], inplace=True)  # essential ID must not be missing
df_health.fillna({'Feedback_Text': '', 'Sentiment_Score': 0}, inplace=True)

# 3. Convert date columns to datetime
df_health['Last_Checkup_Date'] = pd.to_datetime(df_health['Last_Checkup_Date'], errors='coerce')

# 4. Fix data types if needed (example: categorical columns)
df_health['Gender'] = df_health['Gender'].astype('category')
df_health['Risk_Label'] = df_health['Risk_Label'].astype('category')
df_health['Sentiment_Label'] = df_health['Sentiment_Label'].astype('category')
df_health['Followup_Required'] = df_health['Followup_Required'].astype('category')

# 5. Filter out invalid numeric values if any (example: negative ages)
df_health = df_health[df_health['Age'] >= 0]

# --- Cleaning Urban Bengaluru Dataset ---

# 1. Drop duplicates
df_urban.drop_duplicates(inplace=True)

# 2. Handle missing values
df_urban.dropna(subset=['Request_ID'], inplace=True)  # essential ID
df_urban['Complaint_Text'].fillna('', inplace=True)
df_urban.fillna({'Sentiment_Score': 0, 'Delay_Minutes': 0}, inplace=True)

# 3. Convert date columns
df_urban['Created_At'] = pd.to_datetime(df_urban['Created_At'], errors='coerce')

# 4. Fix data types for categorical columns
cat_cols_urban = ['Category', 'Status', 'Neighborhood_Name', 'Sentiment_Label', 'Delay_Severity', 'Day_of_Week']
for col in cat_cols_urban:
    df_urban[col] = df_urban[col].astype('category')

# 5. Remove rows with invalid geographic coordinates (latitude, longitude)
df_urban = df_urban[(df_urban['Latitude'] >= -90) & (df_urban['Latitude'] <= 90)]
df_urban = df_urban[(df_urban['Longitude'] >= -180) & (df_urban['Longitude'] <= 180)]

# --- Save cleaned datasets to desktop folder ---

output_health_file = os.path.join(desktop_path, 'cleaned_predictive_healthcare_dataset.csv')
output_urban_file = os.path.join(desktop_path, 'cleaned_urban_bengaluru_dataset.csv')

df_health.to_csv(output_health_file, index=False)
df_urban.to_csv(output_urban_file, index=False)

print(f"Cleaned Predictive Healthcare dataset saved to: {output_health_file}")
print(f"Cleaned Urban Bengaluru dataset saved to: {output_urban_file}")
