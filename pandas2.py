#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 


# In[5]:


df = pd.read_csv("universities.csv")
df


# In[6]:


df[df["GradRate"]>=95]


# In[13]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[17]:


df.sort_values(ascending=False,by="SFRatio")


# In[18]:


df.sort_values(by="SFRatio")


# In[19]:


sal = pd.read_csv("salaries.csv")
sal


# In[28]:


sal["salary"].groupby(sal["rank"]).mean()


# In[29]:


sal["salary"].groupby(sal["rank"]).sum()


# In[30]:


sal["salary"].groupby(sal["rank"]).median()


# In[25]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[26]:


sal[["salary","phd","service"]].groupby(sal["rank"]).sum()


# In[31]:


sal[["salary","phd","service"]].groupby(sal["rank"]).median()


# In[1]:


import pandas as pd
# Create a sample table
data = {
    'User ID': [1, 2, 2, 3],
    'Movie Name': ['Pushpa2', 'Game Changer', 'Daku Maharaj', 'Pushpa2-Reloaded'],
    'Rating': [9, 3, 8, 9.5]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create pivot table
pivot_table = df.pivot(index='User ID', columns='Movie Name', values='Rating')
print(pivot_table)


# In[ ]:




