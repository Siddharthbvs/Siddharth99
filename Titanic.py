#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("Titanic.csv")
data


# In[3]:


#Install mlxtend libreary
get_ipython().system('pip install mlxtend')


# In[4]:


#Import necessary libraries
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[5]:


#Print the dataframe
titanic = pd.read_csv("Titanic.csv")
titanic


# # observations
# - All columns are categorical
# - No NUll values
# - As the columns are categorical, we adopt one-hot-encoding

# In[6]:


titanic.info()


# In[7]:


#plot a bar chart to visualize the category of class on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[8]:


#plot a bar chart to visualize the category of Gender on the ship
counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[9]:


#plot a bar chart to visualize the category of Age on the ship
counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[10]:


#plot a bar chart to visualize the category of Survived  on the ship
counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[11]:


#preform one-hot encoding on categorical columns
df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[12]:


df.info()


# ### Apriori Algorithm

# In[13]:


#Apply Apriori Alhorithm to get iteset combinations
frequent_itemsets = apriori(df, min_support = 0.05, use_colnames=True, max_len=None)
frequent_itemsets


# In[14]:


#Generate association rules with metrics
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[15]:


rules.sort_values(by='lift', ascending =False)


# ### Colclusion 
# - Adult Females travelling in  1st class were among the most survived

# In[16]:


rules.sort_values(by='lift', ascending =True)


# In[24]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift',]].hist(figsize=(15,7))
plt.show()


# ### Observation
# - In Support value frequency more than 80
# - In Confidence value frequency less than 40
# - In Lift value frequency id more than 100
# 

# In[18]:


import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel("support")
plt.ylabel("support")
plt.show()


# ### Observation
# * The confidence value is increasing with ince=rease in the support

# In[19]:


plt.scatter(rules['confidence'], rules['lift'])
plt.show()


# In[22]:


rules[rules["consequents"]== ({"Survived_Yes"})]


# In[ ]:





# In[ ]:




