# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 16:10:11 2025

@author: aleja
"""

######################### Predict Actuarial Loss #########################

# 1. Exploratory Data Analysis ............... line(13-55)
# 2. Interaction plots         ............... line(75-120)
# 3. Features Correlation      ............... line(75-120)
# 4. Prediction                ............... line(75-120)



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import re
import gc
from IPython.display import display
from tqdm import tqdm, tqdm_notebook
import plotly.express as px
from plotly.offline import iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline = False, world_readable = True)

import plotly.io as pio
pio.templates.default = 'plotly_white'

import itertools
import collections
from collections import Counter

from nltk.corpus import stopwords

import re
from wordcloud import WordCloud

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.style.use('seaborn-whitegrid')
sns.set_palette('Set2')

from time import time, strftime, gmtime
start = time()
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

os.chdir("C:/Users/aleja/Desktop/Travail/JOBS/PROJECTS/Actuarial")
Data = pd.read_csv('x_trn.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


##### 1. Exploratory Data Analysis

#Target Distribution

plt.figure(figsize = (16, 10))
plt.subplot(1, 2, 1)
sns.distplot(train['UltimateIncurredClaimCost'])
plt.subplot(1, 2, 2)
plt.title('Log Scale')
sns.distplot(np.log1p(train['UltimateIncurredClaimCost']));

# Check the missing values in dataset

missing = train.isna().sum().reset_index()
missing.columns = ['features', 'total_missing']
missing['percent'] = (missing['total_missing'] / len(train)) * 100
missing.index = missing['features']
del missing['features']

missing['total_missing'].iplot(kind = 'bar', 
                               title = 'Missing Values Plot in Trainset',
                               xTitle = 'Features',
                               yTitle = 'Count')
missing.T

missing = test.isna().sum().reset_index()
missing.columns = ['features', 'total_missing']
missing['percent'] = (missing['total_missing'] / len(test)) * 100
missing.index = missing['features']
del missing['features']

missing['total_missing'].iplot(kind = 'bar', 
                               title = 'Missing Values Plot in Testset',
                               xTitle = 'Features',
                               yTitle = 'Count')
missing.T

# Marital Status

plt.suptitle('Countplot of Marital Status: M - Married; S - Single; U - Unknown')
plt.subplot(1, 2, 1)
train['MaritalStatus'].value_counts(dropna = False).plot(kind = 'bar', rot = 0);

plt.subplot(1, 2, 2)
train['MaritalStatus'].value_counts(dropna = False).plot(kind = 'bar', rot = 0);

# Date Features

train['DateTimeOfAccident'] = pd.to_datetime(train['DateTimeOfAccident'])
train['DateReported'] = pd.to_datetime(train['DateReported'])

test['DateTimeOfAccident'] = pd.to_datetime(test['DateTimeOfAccident'])
test['DateReported'] = pd.to_datetime(test['DateReported'])

# Age

print('Train:')
print(f"Max. Age: {train['Age'].max()}")
print(f"Min. Age: {train['Age'].min()}")
print(f"Avg. Age: {round(train['Age'].mean(), 2)}")

print('Test:')
print(f"Max. Age: {test['Age'].max()}")
print(f"Min. Age: {test['Age'].min()}")
print(f"Avg. Age: {round(test['Age'].mean(), 2)}")

plt.suptitle('Distribution of Age')

plt.subplot(1, 2, 1)
sns.distplot(train['Age'], color = '#810f7c')
plt.title('Train')

plt.subplot(1, 2, 2)
sns.distplot(test['Age'], color = '#8c96c6')
plt.title('Test');

# Gender

train['Gender'].value_counts().iplot(kind = 'bar', 
                                    yTitle = 'Count', 
                                    xTitle = 'Gender', 
                                    title = 'Count plot of Gender: M - Male; F- Female; U - Unknown/Unspecified')


# Dependents: Children/Others

train['DependentChildren'].max(), train['DependentChildren'].min(), train['DependentChildren'].median()

plt.subplot(1, 2 , 1)
sns.countplot(train['DependentChildren'])
plt.subplot(1, 2 , 2)
sns.countplot(train['DependentsOther']);

# PartTimeFullTime

train['PartTimeFullTime'].value_counts().iplot(kind = 'bar', 
                                    yTitle = 'Count', 
                                    xTitle = 'Type of Work', 
                                    title = 'Type of Work: F - Full Time;  P - Part Time')

#HoursWorkedPerWeek

print(f"Max. hours worked: {train['HoursWorkedPerWeek'].max()}")
print(f"Min. hours worked: {train['HoursWorkedPerWeek'].min()}")
print(f"Avg. hours worked: {round(train['HoursWorkedPerWeek'].mean(), 2)}")
sns.distplot(train['HoursWorkedPerWeek']);

#DaysWorkedPerWeek

train['DaysWorkedPerWeek'].value_counts().iplot(kind = 'bar', 
                                    yTitle = 'Count', 
                                    xTitle = 'No. of Days Worked', 
                                    title = 'Number of Days Worked Per Week')

#InitialIncurredCalimsCost

plt.figure(figsize = (16, 10))
plt.subplot(1, 2, 1)
sns.distplot(train['InitialIncurredCalimsCost'])
plt.subplot(1, 2, 2)
plt.title('Log Scale')
sns.distplot(np.log1p(train['InitialIncurredCalimsCost']));

print('Train:')
print(f"Max. Initial Claim: {train['InitialIncurredCalimsCost'].max()}")
print(f"Min. Initial Claim: {train['InitialIncurredCalimsCost'].min()}")
print(f"Avg. Initial Claim: {round(train['InitialIncurredCalimsCost'].mean(), 2)}")

print('Test:')
print(f"Max. Initial Claim: {test['InitialIncurredCalimsCost'].max()}")
print(f"Min. Initial Claim: {test['InitialIncurredCalimsCost'].min()}")
print(f"Avg. Initial Claim: {round(test['InitialIncurredCalimsCost'].mean(), 2)}")

#Weekly Wages

plt.title('Distribution of Weekly Wages')
sns.distplot(train['WeeklyWages']);

train['WeeklyWages'].max(), train['WeeklyWages'].min(), train['WeeklyWages'].mean()

#Accidents by day of week

train['Acc_Day'].value_counts().iplot(kind = 'bar', 
                                    yTitle = 'Count', 
                                    xTitle = 'Days of Week', 
                                    title = 'Accidents Numbers by Day of Week')

#Accidents by month

train['Acc_Year'].value_counts().iplot(kind = 'bar', 
                                    yTitle = 'Count', 
                                    xTitle = 'Year', 
                                    title = 'Accidents Numbers by Year')

#TimeDiff_Hrs

print('Train:')
print(f"Max. Time Diff in Hrs: {train['TimeDiff_Hrs'].max()}")
print(f"Min. Time Diff in Hrs: {train['TimeDiff_Hrs'].min()}")
print(f"Avg. Time Diff in Hrs: {round(train['TimeDiff_Hrs'].mean(), 2)}")

print('Test:')
print(f"Max. Time Diff in Hrs: {test['TimeDiff_Hrs'].max()}")
print(f"Min. Time Diff in Hrs: {test['TimeDiff_Hrs'].min()}")
print(f"Avg. Time Diff in Hrs: {round(test['TimeDiff_Hrs'].mean(), 2)}")

sns.distplot(train['TimeDiff_Hrs']);

##### 2. Interaction plots

# Weekly ages vs age

px.scatter(train, x = 'Age', y = 'WeeklyWages', size = 'InitialIncurredCalimsCost', 
           color = 'Gender',
          template = 'plotly_white')

# Initial incurred calms cost vs Weekly ages

px.scatter(train, x = 'WeeklyWages', y = 'InitialIncurredCalimsCost', size = 'Age', 
           color = 'PartTimeFullTime',
          template = 'plotly_white')


# Claim vs marital status
pd.pivot_table(data = train, index = 'Gender', 
               columns = ['MaritalStatus'], 
               values = ['InitialIncurredCalimsCost', 'UltimateIncurredClaimCost']).iplot(kind = 'bar', 
                                                                                         xTitle = 'Gender', 
                                                                                         yTitle = 'Claims Cost', 
                                                                                         title = 'Claims by MaritalStatus')

# 3. Features Correlation

corr1 = train[numerical_features].corr(method = 'pearson')

fig = plt.figure(figsize = (10, 8))
#mask = np.triu(np.ones_like(corr1, dtype = bool))
sns.heatmap(corr1, mask = None, annot = True, cmap = 'PiYG', vmin = -1, vmax = +1)
plt.title('Pearson Correlation')
plt.xticks(rotation = 90)
plt.show()

def plot_wordcloud(data, col, text = None):
    stop = stopwords.words('english')
    all_words = [word for each in data[col] for word in str(each).lower().split() if word not in stop]
    word_freq = Counter(all_words)

    wordcloud = WordCloud(width = 900,
                          height = 500,
                          max_words = 200,
                          max_font_size = 100,
                          relative_scaling = 0.5,
                          background_color = "rgba(255, 255, 255, 0)", 
                          mode = "RGBA",
                          normalize_plurals = True).generate_from_frequencies(word_freq)
    plt.figure(figsize = (16, 12))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.title(text, fontsize = 16)
    plt.axis("off")
    plt.show()

plot_wordcloud(train, 'ClaimDescription', 'WordCloud of Claim Description')

# 4. Prediction

from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(
               objective = 'regression', 
               num_leaves = 4,
               learning_rate = 0.01, 
               n_estimators = 10000,
               max_bin = 200, 
               bagging_fraction = 0.75,
               bagging_freq = 5, 
               bagging_seed = 7,
               feature_fraction = 0.2,
               feature_fraction_seed = 7,
               verbose = 1,
            )

lgbm_model = lgbm.fit(Xtrain, ytrain)
lg_vpreds = lgbm_model.predict(Xvalid)
print((f"LGBM RMSE: {np.sqrt(mean_squared_error(yvalid, lg_vpreds))}"))


lg_preds = lgbm_model.predict(test)
sub['UltimateIncurredClaimCost'] = np.expm1(lg_preds)
sub.to_csv('submission.csv', index = False)
sub.head()

from xgboost import XGBRegressor

xgb = XGBRegressor(
                    learning_rate = 0.01, 
                    n_estimators = 10000,
                    max_depth = 3, 
                    min_child_weight = 0,
                    gamma = 0, 
                    subsample = 0.7,
                    colsample_bytree = 0.7,
                    objective = 'reg:squarederror', 
                    nthread = 1,
                    scale_pos_weight = 1, 
                    seed = 27,
                    reg_alpha = 0.00006
                    )
xgb_model = xgb.fit(Xtrain, ytrain)
xg_vpreds = xgb_model.predict(Xvalid)
print((f"XGBOOST RMSE: {np.sqrt(mean_squared_error(yvalid, xg_vpreds))}"))

xg_preds = xgb_model.predict(test)
sub['UltimateIncurredClaimCost'] = np.expm1(xg_preds)
sub.to_csv('submission_xg.csv', index = False)
sub.head()

sub['UltimateIncurredClaimCost'] = np.expm1(lg_preds) + np.expm1(xg_preds)
sub.to_csv('submission_en.csv', index = False)
sub.head()


                                                                                                                                                                      