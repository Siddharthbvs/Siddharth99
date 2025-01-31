#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[5]:


data1.info()


# Observations
# * No null Values are in the data

# In[6]:


data1.describe()


# In[21]:


#Extract outliers from boxplot for daily column
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["daily"],vert=False)
[item.get_xdata() for item in boxplot_data ['fliers']]


# In[33]:


boxplot_data = plt.boxplot(data1["daily"],vert=False)


# In[22]:


sns.kdeplot(data=data1["daily"],fill=True,color="green")


# In[23]:


sns.histplot(data1["daily"], kde=True,color='purple')


# In[24]:


sns.histplot(data1["daily"], kde=False,color='purple')


# In[26]:


plt.scatter(data1["daily"],data1["sunday"])


# Observation
# * A high correlation has been observed

# In[31]:


data1["daily"].corr(data1["sunday"])


# In[47]:


#Build regression model
import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[49]:


model.summary()


# In[46]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
# predicted response vector
y_hat = b0 + b1*x
# plotting the regression line
plt.plot(x, y_hat, color
= "g")
# putting Labels
plt.xlabel('x')
plt.ylabel('y')

plt.show()


# In[ ]:




