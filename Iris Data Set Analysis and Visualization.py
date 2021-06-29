#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from sklearn.datasets import load_iris
Iris = load_iris()


# In[6]:


Iris


# In[7]:


Iris.feature_names


# In[8]:


Iris.target_names


# In[9]:


df=pd.DataFrame(Iris.data, columns= Iris.feature_names)
df['Target']= Iris.target


# In[10]:


df.describe()


# ##  Visualization

# In[12]:


df.plot(kind='box', subplots=True, layout=(3,2), figsize=(8,12));


# In[14]:


df.hist(figsize= (12,12))
plt.show()


# In[16]:


df.corr()


# In[17]:


sns.heatmap(df.corr(), annot=True, cmap='Wistia')


# In[18]:


sns.pairplot(df)


# In[23]:


sns.catplot(x='petal length (cm)',y= 'petal width (cm)',palette='husl', hue= 'Target', data=df)


# In[ ]:





# In[ ]:




