#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)

data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[5]:


#drop dupplication column (temp c ) and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis = 1)
data1


# In[6]:


#convert th month column data type to float data type
data1['Month'] = pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[8]:


#checking for duplicated rows in the table
#print the duplicated row(one)only
data[data1.duplicated()]


# In[9]:


#drop duplicated data
data1.drop_duplicates(keep='first', inplace =True)
data1


# In[10]:


#chhange column names (Rename the columns)
data1.rename({'Solar.R': 'Solar'},axis=1, inplace = True)
data1


# In[11]:


data1.info()


# In[14]:


#display data1 missing values count in eachcolumn using is null().sum()
data1.isnull().sum()


# In[19]:


#visualize data1 missing values graph
cols = data1.columns
colors = ['yellow','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[21]:


#find the meanand median values of each numeric column
#Imputation of missing values with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[22]:


#Replace the ozone missing values with median values
data1["Ozone"] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[24]:


#find the meanand median values of each numeric column
#Imputation of missing values with median
median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ",median_solar)
print("Mean of Solar: ",mean_solar)


# In[26]:


#Replace the solar missing values with median values
data1["Solar"] = data1['Solar'].fillna(mean_solar)
data1.isnull().sum()


# In[28]:


#print the data1 5 rows
data1.head()


# In[33]:


#find the mode values of categorical columns weather
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[36]:


#impute misssing values (replace nan with mode etc)using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:





# In[ ]:




