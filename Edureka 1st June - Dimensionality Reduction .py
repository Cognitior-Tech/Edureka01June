#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('D:\\Edureka\\08APR - 2019 - Python\\Misc\\PCA\\')


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


dataset = pd.read_csv('Bank-data.csv')


# In[4]:


dataset


# In[5]:


dataset['Churn'] = dataset['Churn'].astype('category')


# In[6]:


dataset['Churn'] = dataset['Churn'].cat.codes


# In[7]:


dataset


# In[10]:


x = dataset.iloc[:,0:6].values


# In[12]:


y = dataset.iloc[:,6].values


# In[13]:


y


# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[15]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)


# In[16]:


y_pred = logmodel.predict(x_test)


# In[30]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train_n, y_train)


# In[31]:


y_pred = logmodel.predict(x_test_n)


# In[32]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
#89.42%


# In[20]:


from sklearn.decomposition import PCA #Principal component analysis
pca = PCA(n_components=None)
x_train_n = pca.fit_transform(x_train)
x_test_n = pca.fit_transform(x_test)


# In[28]:


from sklearn.decomposition import PCA #Principal component analysis
pca = PCA(n_components=2)
x_train_n = pca.fit_transform(x_train)
x_test_n = pca.fit_transform(x_test)


# In[22]:


pd.DataFrame(x_train)


# In[29]:


pd.DataFrame(x_train_n)


# In[25]:


explained_variance = pca.explained_variance_ratio_


# In[33]:


pd.DataFrame(explained_variance)

