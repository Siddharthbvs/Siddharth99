#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# Clustring- Divide the universities into groups(clusters

# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


Univ.isna().sum()


# In[5]:


Univ.describe()


# Standardization of the data

# In[6]:


#Read all the numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[7]:


Univ1


# In[8]:


cols =Univ1.columns


# In[9]:


#Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[10]:


#Build 3 Cluster using KMeans Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
#Specify 3 clusters
clusters_new.fit(scaled_Univ_df)


# In[11]:


#print the clusters labels
clusters_new.labels_


# In[12]:


set(clusters_new.labels_)


# In[13]:


#Assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[14]:


Univ


# In[15]:


Univ[Univ['clusterid_new']==1]


# In[16]:


#Use groupby to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### obsevations
# - cluster 1 appears to be the top rated Universities cluster as the cut off score, Top10, SF Ratio parameter mean values are highest
# - Cluster 2 appears to occupy the middel level rated Universities
# - Cluster 0 appears to occupy Lower level rated Universities

# ## Finding optimal k value using elbow plot

# In[24]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # Find the quality of clusters

# In[23]:


# Quality of clusters is expresssed in terms of silhoutte score
from sklearn.metrics import silhouette_score
score = silhouette_score(scaled_Univ_df,clusters_new.labels_, metric='euclidean')
score


# In[ ]:




