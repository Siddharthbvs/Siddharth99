#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd

df = pd.read_csv('Day_13_Pharma_data.csv')
print(df.isnull().sum())
print(df.duplicated().sum())

df = df.drop_duplicates()  
df = df.fillna(df.mean())
print(df.info())


# In[20]:


import matplotlib.pyplot as plt
region_sales = df.groupby('Region')['Sales'].sum()

region_sales.plot(kind='bar', color='lightblue', title='Total Sales per Region')

plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)  # Rotate the x-axis labels if needed
plt.show()


# In[21]:


import seaborn as sns

sns.scatterplot(x='Marketing_Spend', y='Sales', data=df, color='purple')
plt.title('Marketing Spend vs Sales')
plt.xlabel('Marketing Spend')
plt.ylabel('Sales')
plt.show()


# In[22]:


sns.boxplot(x='Age_Group', y='Effectiveness', data=df, palette='Set2')

plt.title('Drug Effectiveness Across Different Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Effectiveness')
plt.show()


# In[23]:


corr_matrix = df[['Sales', 'Marketing_Spend', 'Effectiveness']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Heatmap: Sales, Marketing Spend, Effectiveness')
plt.show()


# In[ ]:




