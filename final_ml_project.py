# ==========================================
# Codomax AI & Python Internship
# Final AI & ML Capstone Project: Customer Churn Predictor
# Student Name: M. HAZAIFA TAHIR
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ------------------------------------------
# 1. GENERATING SYNTHETIC DATASET (Self-contained)
# ------------------------------------------
print("Generating synthetic Customer Churn dataset...")
np.random.seed(42)
n_samples = 1500

data = {
    'CustomerID': range(1001, 1001 + n_samples),
    'Age': np.random.randint(18, 70, size=n_samples),
    'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
    'Tenure': np.random.randint(1, 72, size=n_samples), # months with company
    'MonthlyCharges': np.random.uniform(20.0, 120.0, size=n_samples),
    'TotalCharges': np.zeros(n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.5, 0.3, 0.2]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], size=n_samples)
}

# TotalCharges usually correlates with Tenure and MonthlyCharges
for i in range(n_samples):
    data['TotalCharges'][i] = data['Tenure'][i] * data['MonthlyCharges'][i] * np.random.uniform(0.9, 1.1)

df = pd.DataFrame(data)

# Injecting logic for Churn (Target Variable)
# Customers with high charges, low tenure, or month-to-month contracts are more likely to churn.
churn_prob = (df['MonthlyCharges'] / 120.0) * 0.4 + (1 - (df['Tenure'] / 72.0)) * 0.5
churn_prob += (df['Contract'] == 'Month-to-month').astype(int) * 0.3
df['Churn'] = (churn_prob > 0.8).astype(int)

print(f"Dataset Shape: {df.shape}")
print(f"Churn Distribution:\n{df['Churn'].value_counts(normalize=True)}")

# ------------------------------------------
# 2. DATA PREPROCESSING & CLEANING
# ------------------------------------------
print("\nPreprocessing data...")
# Copy dataset to preserve original
df_proc = df.copy()

# Drop identifiers
df_proc.drop(columns=['CustomerID'], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'Contract', 'InternetService', 'PaperlessBilling']

for col in categorical_cols:
    le = LabelEncoder()
    df_proc[col] = le.fit_transform(df_proc[col])
    label_encoders[col] = le

# Handle missing values if any
df_proc['TotalCharges'] = df_proc['TotalCharges'].fillna(df_proc['TotalCharges'].median())

# Train-Test Split
X = df_proc.drop(columns=['Churn'])
y = df_proc['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------
# 3. TRAINING ML MODEL (Random Forest)
# ------------------------------------------
print("\nTraining Random Forest Classifier on Customer Data...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# ------------------------------------------
# 4. MODEL EVALUATION
# ------------------------------------------
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- MODEL RESULTS ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print(f"ROC AUC Score : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot and save feature importance
importances = model.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Feature Importances in Churn Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Importance Value')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance graph saved as 'feature_importance.png'.")

# ------------------------------------------
# 5. TEST USER PREDICTION (Inference)
# ------------------------------------------
def predict_new_customer(age, gender, tenure, monthly_charges, contract, internet_service, paperless_billing):
    gender_enc = label_encoders['Gender'].transform([gender])[0]
    contract_enc = label_encoders['Contract'].transform([contract])[0]
    internet_enc = label_encoders['InternetService'].transform([internet_service])[0]
    paperless_enc = label_encoders['PaperlessBilling'].transform([paperless_billing])[0]
    
    total_charges = tenure * monthly_charges
    
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_enc,
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract_enc,
        'InternetService': internet_enc,
        'PaperlessBilling': paperless_enc
    }])
    
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    churn_class = model.predict(input_scaled)[0]
    
    result = "CHURN (Highly likely to leave)" if churn_class == 1 else "STAY (Loyal customer)"
    return result, prob

# Test the prediction logic
print("\nPredicting status for sample risk customer:")
status, probability = predict_new_customer(
    age=34, gender='Female', tenure=3, monthly_charges=110.50, 
    contract='Month-to-month', internet_service='Fiber optic', paperless_billing='Yes'
)
print(f"Result: {status} (Probability: {probability * 100:.2f}%)")
