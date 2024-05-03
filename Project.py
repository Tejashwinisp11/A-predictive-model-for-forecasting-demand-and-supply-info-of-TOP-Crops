#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[16]:


data = pd.read_csv(r"C:\Users\shilp\Downloads\Final_Training_Data.csv")


# In[28]:


print(data)


# In[17]:


data.head()


# In[18]:


df.dtypes


# In[19]:


df["gender"].unique()


# In[20]:


df["residence"].unique()


# In[21]:


df["location"].unique()


# In[27]:


# Perform one-hot encoding
data_encoded = pd.get_dummies(data)
print(data_encoded)
data_encoded.columns


# In[23]:


# Segregating Features & Target labels
X = data_encoded.drop(columns=['TARGET_PREDICTION_PERCENT'])
y = data_encoded['TARGET_PREDICTION_PERCENT']


# In[24]:


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[26]:


# Print the first few rows of the transformed training data
print("Transformed Training Data:")
print(X_train_scaled[:5])

