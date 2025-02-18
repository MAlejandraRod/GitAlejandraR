# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:19:24 2025

@author: aleja
"""
######################### AD CAMPAIGN PERFORMANCE ############################

# 1. Exploratory Data Analysis ............... line(16-68)
# 2. Key Performance Metrics
# 3. Performance Analysis
# 4. Segmentation & Optimizations
# 5. Recomendations & next setps


##### 1. Exploratory Data Analysis 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("C:/Users/aleja/Desktop/Travail/JOBS/PROJECTS/ad_campaign_performance")
Data = pd.read_csv('ad_campaign_performance.csv')

# Summary: The budget allocation is varied and is not concentrated in a single segment,  
# Youtube (226 campaings) is the most used platform, Story (213 campaings) is the most used content type,
# People from 35 to 44 (217 campaings) is the main target, All genders (346 campaings) is the main target,
# UK (224) is the main target region.

# No duplicated ID

print(Data['Campaign_ID'].duplicated(keep=False).sum()) # 0 duplicated

# No missing values

print(Data.isnull().sum())

# Budget distribution is heterogeneous: The budget allocation is varied and is 
# not concentrated in a single segment.

plt.hist(Data['Budget'], bins=20, color='purple', alpha=0.7, edgecolor='black')

# Duration distribution is heterogeneous: The duration allocation is varied.

plt.hist(Data['Duration'], bins=20, color='blue', alpha=0.7, edgecolor='black')

# Youtube (226 campaings) is the most used platform followed by LikedIn (209) and 
# Instagram (202). The less used is Google (170).

print(Data['Platform'].value_counts())

# Story (213 campaings) is the most used content type followed by Image (210) and 
# Carousel (202). The less used is videos (175).

print(Data['Content_Type'].value_counts())

# People from 35 to 44 (217 campaings) is the main target followed by ages from 55+ (205) and 
# 18-24 (202). The less range target is 45-54 (184).

print(Data['Target_Age'].value_counts())

# All (346 campaings) is the main target followed by female (337) and male (317).

print(Data['Target_Gender'].value_counts())

# UK (224) is the main target region, followed by US (204) and Germany (197). The last region
# is Canada (186).

print(Data['Region'].value_counts())


##### 2. Key Performance Metrics KPI

# Conversion Rate: The percentage of users who clicked and then completed the 
# desired action.
Data['Conversion_Rate'] = Data.apply(lambda row: (row['Conversions'] / row['Clicks']) * 100 if row['Clicks'] > 0 else 0, axis=1)

# Cost per Click: amount an advertiser pays each time a user clicks on their ad. 
# Lower CPC is getting more clicks for your budget.
Data['CPC'] = Data.apply(lambda row: row['Budget'] / row['Clicks'], axis = 1)

# Cost per Conversion: how much you spend per conversion.

Data['CPA'] = Data.apply(lambda row: row['Budget'] / row['Conversions'], axis = 1)

##### 2. Performance Analysis
# Analyze the effectiveness of different factors in the campaign.

### Platform Performance

# YOUTUBE is the platform where:
# - people clicked the most after seeing the ad (CTR)
# - people clicked the most and then completed the desired action (CVR)
# - is paying more for each click on the ad (CPC)
# LINKEDIN is the more efficient platform at converting users at a lower cost (CPA)

top_CTR_per_platform = Data.loc[Data.groupby('Platform')['CTR'].idxmax(),
                                ['Campaign_ID', 'Platform', 'CTR']].set_index('Platform')#Youtube

top_CVR_per_platform = Data.loc[Data.groupby('Platform')['Conversion_Rate'].idxmax(),
                                ['Campaign_ID','Platform','Conversion_Rate']].set_index('Platform')#Youtube

top_CPC_per_platform = Data.loc[Data.groupby('Platform')['CPC'].idxmax(),
                                ['Campaign_ID', 'Platform', 'CPC']].set_index('Platform')#Youtube

top_CPA_per_platform = Data.loc[Data.groupby('Platform')['CPA'].idxmax(),
                                ['Campaign_ID', 'Platform', 'CPA']].set_index('Platform')#LinkedIn


top_perform_per_platform = pd.concat([top_CTR_per_platform,
                                      top_CVR_per_platform[['Conversion_Rate']],
                                      top_CPC_per_platform[['CPC']],
                                      top_CPA_per_platform[['CPA']]], axis=1)

top_perform_per_platform.reset_index(inplace=True)

melted_top_performe = top_perform_per_platform.melt(id_vars = ['Platform'],
                                                    value_vars = ['CTR','Conversion_Rate','CPC','CPA'],
                                                    var_name = 'Metric', value_name= 'Value')

plt.figure(figsize=(12,6))
sns.barplot(data = melted_top_performe, x='Platform', y='Value',hue='Metric')

### Ad content type effectiveness

# Whit STORY people clicked the most after seeing the ad (CTR) but It's paying more for each 
# click (CPC).
# Whit IMAGEs people clicked the most and completed the desired action (CVR) and 
# are more efficient at converting users at a lower cost (CPC). The less efective are VIDEOs.


top_CTR_per_ContentType = Data.loc[Data.groupby('Content_Type')['CTR'].idxmax(),
                                   ['Campaign_ID','Content_Type',
                                    'CTR']].set_index('Content_Type')#Story

top_CVR_per_ContentType = Data.loc[Data.groupby('Content_Type')['Conversion_Rate'].idxmax(),
                                   ['Campaign_ID', 'Content_Type',
                                    'Conversion_Rate']].set_index('Content_Type')#Image

top_CPC_per_ContentType = Data.loc[Data.groupby('Content_Type')['CPC'].idxmax(),
                                   ['Campaign_ID', 'Content_Type',
                                    'CPC']].set_index('Content_Type')#Story

top_CPA_per_ContentType = Data.loc[Data.groupby('Content_Type')['CPA'].idxmax(),
                                   ['Campaign_ID', 'Content_Type',
                                    'CPA']].set_index('Content_Type')#Video

top_perform_per_ContentType = pd.concat([top_CTR_per_ContentType,
                                        top_CVR_per_ContentType[['Conversion_Rate']],
                                        top_CPC_per_ContentType[['CPC']],
                                        top_CPA_per_ContentType[['CPA']]], axis=1)


