#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 


# In[4]:


df = pd.read_csv("universities.csv")
df


# In[5]:


#mean
np.mean(df["SAT"])


# In[6]:


#median
np.median(df["SAT"])


# In[8]:


#standard deviation of data 
np.std(df["GradRate"])


# In[9]:


#find variance
np.var(df["SFRatio"])


# In[11]:


df.describe()


# In[ ]:




