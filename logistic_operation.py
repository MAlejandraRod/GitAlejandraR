# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:41:30 2025

@author: aleja
"""

######################### LOGISTIC OPERATION ############################

# 1. Exploratory Data Analysis ............... line(14-74)
# 2. Predictive Modeling       ............... line(75-120)

##### 1. Exploratory Data Analysis 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


os.chdir("C:/Users/aleja/Desktop/Travail/JOBS/PROJECTS/Logistics_operation")
Data = pd.read_csv('smart_logistics_dataset.csv')

Data['Timestamp'] = pd.to_datetime(Data['Timestamp'])

# Check for missing values
missing_values = Data.isnull().sum()
print('Missing values in each column:')
print(missing_values)

# Histogram of Inventory Level: the warehouse might be maintaining stable inventory 
# replenishment without major fluctuations.

sns.histplot(Data['Inventory_Level'], kde=True, color='skyblue')
plt.title('Distribution of Inventory Level')
plt.xlabel('Inventory Level')
plt.ylabel('Frequency')
plt.show()

# potential logistics or operational issues are causing delays, many shipments eventually 
# reach their destination, but there might be some inefficiencies or bottlenecks in 
# transitioning from In Transit to Delivered. There are signal of supply chain inefficiencies, 
# logistical issues, or unexpected disruptions.

sns.countplot(data=Data, x='Shipment_Status', palette='viridis')
plt.title('Shipment Status Count')
plt.xlabel('Shipment Status')
plt.ylabel('Count')
plt.show()

# Correlation among variables: most correlations are very weak (close to 0), indicating 
# that these variables do not have strong linear relationships with each other.

numeric_df = Data.select_dtypes(include=[np.number])

if numeric_df.shape[1] >= 4:
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()
else:
    print('Not enough numeric columns for a correlation heatmap.')

# pairwise relationships: The absence of strong linear relationships suggests that 
# machine learning techniques (e.g., decision trees, clustering, or neural networks) 
# might be more effective than simple linear regression.
sns.pairplot(numeric_df, diag_kind='kde', corner=True)
plt.show()

##### 2. Predictive Modeling for delay status: Random forest is the best model to predict when 
#the status its going to be "delay" or not. Shipment status and traffic Sattus are the most important features for the prediction.

target = 'Logistics_Delay'
features_to_drop = ['Timestamp', 'Asset_ID', target]

X = Data.drop(features_to_drop, axis=1) #features
y = Data[target] #target variable

# For categorical variables, we use one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

print('Feature matrix shape:', X_encoded.shape)
print('Target vector shape:', y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, 
                                                    test_size=0.2, random_state=42)

# RANDOM FOREST
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the performance using R2 score and Root Mean Squared Error
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('R2 Score:', r2)
print('RMSE:', rmse)

#Feature Importance Analysis: Shipment status and traffic Sattus are the most important
#features to make the prediction.

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_encoded.columns

plt.figure()
plt.barh(np.array(features)[indices], importances[indices], color='teal')
plt.xlabel('Feature Importance')
plt.title('Permutation Importance of Features')
plt.gca().invert_yaxis()
plt.show()

