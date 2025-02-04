#!/usr/bin/env python
# coding: utf-8

# # Asssumptipns in multiple Linear Regression
# 1. Linearity:The Relationship b/w the predicators and the response is linear
# 2. Independence:Obsevations are independent of each other.
# 3. Homoscendasticity: The residual(differeces b/w observed and predicated values) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed.
# 5. No Multicolinearity: The Independent variable should not be too highly correleated with ech other.

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


#Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


#rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Descrption of columns
# - HP : Horse power of the car
# - MPG : Millage of the car(Miles Per Gallon)
# - VOL : Volume of the car(Size)
# - SP : Top speed of the car(Miles per hour)
# - WT : Weight of the car

# In[8]:


cars.info()


# In[9]:


#check for missing values
cars.isna().sum()


# # observations
# - There are no missing values 
# - There are 81 obseravtions
# - The dataa types of the columns are relevant and valid

# In[12]:


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


# In[ ]:




