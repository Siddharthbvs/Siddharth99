#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
iris = pd.read_csv('iris.csv')
iris


# In[2]:


iris.info()


# In[22]:


# Bar plot for categorical column "variety'
import seaborn as sns
counts=iris["variety"].value_counts()
sns.barplot(data = counts)


# #### Observations
# * There are no null values
# * There are three flower categories
# * There are 150 rows and 5 columns
# * There are no duplicated values
# * There are one duplicated rows
# * The x-columns are sepal.length,sepal.width,petal.length
# * The y-column is "variety" which is categrorical

# In[23]:


iris.head(3)


# In[24]:


iris["variety"].value_counts()


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[26]:


iris.info()


# In[27]:


iris[iris.duplicated(keep=False)]


# #### Observations
# * There are no null values
# * There are three flower categories
# * There are 150 rows and 5 columns
# * There are no duplicated values
# * There are one duplicated rows
# * The x-columns are sepal.length,sepal.width,petal.length
# * The y-column is "variety" which is categrorical

# In[28]:


# Drop the duplicates
iris = iris.drop_duplicates(keep='first')


# In[29]:


iris[iris.duplicated]


# In[30]:


# Reset the index
iris = iris.reset_index(drop=True)
iris


# In[31]:


#Encode the three flower classes as 0,1,2
labelencoder=LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[32]:


iris.info()


# #### Observation
# * The target column('variety') is still object type.it needs to be converted to numeric(int).

# In[33]:


#Convert the target column data type to integer
label_encoder = LabelEncoder()
iris['variety'] = label_encoder.fit_transform(iris['variety'])
print(iris.info())


# In[34]:


# Divide the dataset into x and y
X=iris.iloc[:,0:4]
Y=iris['variety']


# In[35]:


Y


# In[36]:


iris.head(3)


# In[37]:


#Further splitting of data into training and testing the data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
x_train


# In[38]:


#Building Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion='entropy',max_depth = None)
model.fit(x_train,y_train)


# In[39]:


#Plot the decision tree
plt.figure(dpi=1200)
tree.plot_tree(model);


# In[40]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor','verginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[41]:


#Predicting on test data 
preds = model.predict(x_test) #predicting on test data set
preds


# In[42]:


print(classification_report(y_test,preds))


# In[ ]:




