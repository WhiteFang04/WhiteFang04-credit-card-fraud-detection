#!/usr/bin/env python
# coding: utf-8

# In[2]:


# model_training.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("creditcard.csv")

# Feature scaling
scaler_amount = StandardScaler()
scaler_time = StandardScaler()

df['scaled_amount'] = scaler_amount.fit_transform(df[['Amount']])
df['scaled_time'] = scaler_time.fit_transform(df[['Time']])

df.drop(['Amount', 'Time'], axis=1, inplace=True)
df.rename(columns={'scaled_amount': 'Amount', 'scaled_time': 'Time'}, inplace=True)

# Split data
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_sm, y_train_sm)

# Save model and scalers
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(scaler_amount, 'models/scaler_amount.pkl')
joblib.dump(scaler_time, 'models/scaler_time.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_columns.pkl')

print("Training complete and files saved in 'models/' folder.")




