#!/usr/bin/env python
# coding: utf-8

# # Asssumptipns in multiple Linear Regression
# 1. Linearity:The Relationship b/w the predicators and the response is linear
# 2. Independence:Obsevations are independent of each other.
# 3. Homoscendasticity: The residual(differeces b/w observed and predicated values) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed.
# 5. No Multicolinearity: The Independent variable should not be too highly correleated with ech other.

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


#Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


#rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Descrption of columns
# - HP : Horse power of the car
# - MPG : Millage of the car(Miles Per Gallon)
# - VOL : Volume of the car(Size)
# - SP : Top speed of the car(Miles per hour)
# - WT : Weight of the car

# In[4]:


cars.info()


# In[5]:


#check for missing values
cars.isna().sum()


# # observations
# - There are no missing values 
# - There are 81 obseravtions
# - The dataa types of the columns are relevant and valid

# In[6]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[9]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()            


# In[10]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[11]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# In[12]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots (2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot
#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
#Adjust Layout
plt.tight_layout()
plt.show()


# # Observation
# - There are some extreme values(outliers) observed in towards the right tail of SP and HP distrubitions.
# - In vol and WT mcolumns , a few outliers are observed in both tails of their distributions,
# - The extreme values of car data may have come from the specially designed nature of cars 
# - As this is multi-dimensional data,the outliers with respect to spatial dimensions may have  to be considered while bulding the regression 

# # CHECKING FOR DUBLICATE 

# In[13]:


cars[cars.duplicated()]


# In[14]:


# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[15]:


cars.corr()


# # Observations
# - High Positive Correlation between VOL and WT 
# - Negative Correlation between MPG and other variables
# - Strong Positive Correlation between HP and SP
# - The highest corr among x columns is not desirable as it might lead to multicollinearity problem

# In[19]:


#Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~+VOL+SP+HP',data=cars).fit()


# In[20]:


model1.summary()


# In[ ]:




