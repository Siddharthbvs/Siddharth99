#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd 
import numpy as np 


# In[32]:


df = pd.read_csv("universities.csv")
df


# In[33]:


#mean
np.mean(df["SAT"])


# In[34]:


#median
np.median(df["SAT"])


# In[35]:


#standard deviation of data 
np.std(df["GradRate"])


# In[36]:


#find variance
np.var(df["SFRatio"])


# In[37]:


df.describe()


# In[38]:


#visulize the GradeRate using histigram
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[40]:


sns.histplot(df["Accept"], kde = True)


# In[41]:


sns.histplot(df["Accept"], kde = False)


# In[42]:


#visulization using boxplot
#create a pandas series of batman1 scores
s1 = [50,15,59,99,45,75,85,67,29]
scores1 = pd.Series(s1)
scores1


# In[43]:


plt.figure(figsize=(6,2))
plt.title("Boxplot for batsman scores")
plt.xlabel("Scores")
plt.boxplot(scores1,vert=False)


# In[55]:


#Add exteam vlues to scores
s2 = [50,15,59,99,45,75,85,67,29,150,120,130]
scores2 = pd.Series(s2)
print(scores2)
plt.figure(figsize=(6,2))
plt.title("Boxplot for batsman scores")
plt.xlabel("Scores")
plt.boxplot(scores1,vert=False)


# In[ ]:





# In[ ]:




