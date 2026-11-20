# AI/ML Customer Churn Predictor

**Intern:** M. HAZAIFA TAHIR  
**Organization:** Codomax Digital Solutions  
**Graduation Track:** Final AI & ML Capstone Project  

## Overview
This is the **Final Capstone AI/ML Project** for the Codomax internship. The goal of this application is to predict customer churn (the likelihood of a user leaving a service) based on key behavioral and demographic metrics using an optimized **Random Forest Classifier** implemented in Python.

## Features
* **Data Synthesis**: Automatically simulates a dataset of 1500 custom users tracking variables like Tenure, Monthly charges, Contract type, and Age.
* **Preprocessing Pipeline**: Complete handling of label encoding, train-test splitting, and Feature Scaling (StandardScaler).
* **Machine Learning Classification**: Features training with `Scikit-learn`'s `RandomForestClassifier`.
* **Model Evaluation**: Metrics include test accuracy, Precision, Recall, ROC AUC scores, and automated Confusion Matrix analysis.
* **Insights Extraction**: Includes statistical code assessing feature importances, illustrating the primary drivers of subscriber loss.
* **Interactive Inference**: Implemented function logic allowing users to check survival odds of sample clients on the fly.

## Run Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
