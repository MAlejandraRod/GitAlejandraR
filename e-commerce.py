# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:03:51 2025

@author: aleja
"""

######################### E-COMMERCE DATA ANALYSIS ############################

# 1. Exploratory Data Analysis ............... line(13-55)
# 2. Predictive Model          ............... line(75-120)

##### 1. Exploratory Data Analysis 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

os.chdir("C:/Users/aleja/Desktop/Travail/JOBS/PROJECTS/E-commerce")
Data = pd.read_csv('Ecommerce_Delivery_Analytics_New.csv')

print(Data.info())

### Visualizations

# Histogram Delivery Time (Minutes): normal distribution
plt.figure(figsize=(10, 6))
sns.histplot(Data['Delivery Time (Minutes)'], kde=True, color='blue')
plt.title('Distribution of Delivery Time (Minutes)')
plt.xlabel('Delivery Time (Minutes)')
plt.ylabel('Count')
plt.show()

# Box Plot Order Value: Most orders are under 1000, with a concentration between 250 y 750.
plt.figure(figsize=(10, 6))
sns.boxplot(x=Data['Order Value (INR)'], color='lightgreen')
plt.title('Box Plot of Order Value (INR)')
plt.xlabel('Order Value (INR)')
plt.show()

# Pie Chart Plot for Platform: each platform present a simmilar number of orders around 33000.
#and 
plt.figure(figsize=(10, 6))
sns.countplot(data=Data, x='Platform', palette='pastel')
plt.title('Count of Orders by Platform')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.show()

##### 2. Predictive Model: The best model to do the prediction is random forest and
# the most important variable is service rating.


model_Data = Data.copy()

# Drop unuseful features
drop_cols = ['Order ID', 'Customer ID', 'Order Date & Time', 'Customer Feedback']
model_Data = model_Data.drop(columns=drop_cols)

# Encode categorical variables
le = LabelEncoder()
for col in model_Data.select_dtypes(include=['object']).columns:
    model_Data[col] = le.fit_transform(model_Data[col])

# Define features and target. We will consider predicting Refund Requested
X = model_Data.drop(columns=['Refund Requested'])
y = model_Data['Refund Requested']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set and calculate the accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc}")# Accuracy = 1

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Permutation Importance Plot (basic version)
importances = clf.feature_importances_
indices = np.argsort(importances)
features = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], color='purple', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Permutation Importance')
plt.show()
