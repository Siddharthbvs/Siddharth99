#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
data = {
    'Name': ['John', 'Alice', 'Bob', 'Diana'],
    'Age': [28, 34, 23, 29],
    'Department': ['HR', 'IT', 'Marketing', 'Finance'],
    'Salary': [45000, 60000, 35000, 50000]
}

df = pd.DataFrame(data)


# In[8]:


print("First 2 rows of the DataFrame:")
print(df.head(2))


# In[12]:


df['Bonus'] = df['Salary'] * 0.10


# In[10]:


average_salary = df['Salary'].mean()
print(f"\nAverage Salary: {average_salary}")


# In[11]:


filtered_df = df[df['Age'] > 25]
print("\nEmployees older than 25:")
print(filtered_df)


# In[ ]:




