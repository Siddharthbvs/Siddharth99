#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[25]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[26]:


data1.info()


# Observations
# * No null Values are in the data

# In[27]:


data1.describe()


# In[28]:


data1.isnull().sum()


# CORRELATION

# In[40]:


data1["daily"].corr(data1["sunday"])


# In[41]:


data1[["daily","sunday"]].corr()


# In[42]:


data1.corr(numeric_only=True)


# In[43]:


plt.scatter(data1["daily"], data1["sunday"])


# In[44]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[45]:


import seaborn as sns
sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[47]:


mport statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[48]:


model.summary()


# Observation
# * A high correlation has been observed
# * The relationship between x (daily) and y(sunday) is seen to be linear as seen from scatter plot

# In[35]:


data1["daily"].corr(data1["sunday"])


# # Fit Linear Regression Model

# In[36]:


#Build regression model
import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[37]:


model.summary()


# Observation
# * The predected equation is Y_hat = beta_0 + beta_1+x
# * y_hat = 13.8356 + 1.3397

# In[38]:


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


# Observations
# 
# * There are no missing values
# * The daily column values appears to be right-skewed
# * The sunday column values also appear to be right-skewed
# * There are two outliers in both daily column and also in sunday column as observed from the boxplot.

# In[39]:


x= data1["daily"]
y= data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(x) + 100)
plt.show()


# - The probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# - Therefore the intercept coefficient may not be that much significant in prediction 
# - However the p-value for "daily" (beta_1) is 0.00 < 0.05
# - Therefore the beta_1 coefficient is highly significant and is contributint to prediction.

# In[53]:


#plot the linear regression line seaborn regplot() mrthod
sns.regplot(x="daily", y="sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# In[ ]:




