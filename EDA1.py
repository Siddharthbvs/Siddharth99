#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("data_clean.csv")
data


# In[5]:


data.info()


# In[8]:


print(type(data))
print(data.shape)
print(data.size)

data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[7]:


#drop dupplication column (temp c ) and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis = 1)
data1


# In[9]:


#convert th month column data type to float data type
data1['Month'] = pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[10]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[12]:


#checking for duplicated rows in the table
#print the duplicated row(one)only
data[data1.duplicated()]


# In[14]:


#drop duplicated data
data1.drop_duplicates(keep='first', inplace =True)
data1


# In[ ]:




