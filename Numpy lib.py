#!/usr/bin/env python
# coding: utf-8

# # Numpy 
# It provide multidimensional array object and tools for working with these array
# array is a data structure that store value of same data type this is the main difference between array and the list

# In[1]:


import numpy as np


# In[ ]:


my_1st = [1,2,3,4,5]
my_1st
arr = np.array(my_1st)
type(arr) # type is a bulitin function
arr.shape
arr.reshape()
arr[3]  # index and slicing
arr[:,:]
arr[1:,0:]
arr[1:2,0:2]
arr = np.arange(0,10)
arr = np.arange(0,10,step=2)
np.linspace(1,10,50) 
arr1=arr # reference type
arr1=arr.copy()  # copy function is used to copy and broadcasting
print(arr)  
arr1[3:]=1000
print(arr1)
val=2     # some condition
arr<2
arr/2 # we can *, / , % ,// with the arr
arr[arr<2]
arr[arr<100]
arr1=np.arange(0,10).reshape(5,2)
arr2=np.arange(0,10).reshape(5,2)
arr1*arr2
arr=np.ones((4,4),dtype=int)
arr = np.random.rand(3,3)
arr = np.random.randn(3,3)
sns.distplot(pd.DataFrame(arr.reshape(16,1)))
np.random.randint(0,100,8).reshape(4,2) # range between 0 and 100 8 element select randomly and reshape it as we want
np.random.random_sample((1,5)) # return a float value between 0 and 1


# In[8]:


# multidimonsional array
arr1 = [1,2,3,4,5]
arr2 = [6,7,8,9,10]
arr3 = [11,12,13,14,15]


# In[10]:


arr = np.array([arr1,arr2,arr3])


# In[11]:


arr


# In[15]:


arr[1:,2:4]


# In[19]:


arr[1:2,1:4]


# In[ ]:





# In[13]:


arr.shape


# In[14]:


arr.reshape(5,3)


# In[21]:


arr  =np.arange(0,10)
arr


# In[22]:


arr  =np.arange(0,10,step = 2)
arr


# In[23]:


arr = np.linspace(1,10,50)
arr


# In[24]:


# copy function and broad casting
arr


# In[25]:


arr = np.array([1,2,3,4,5,6,7,8,9])
arr


# In[26]:


arr[3:]=100
arr


# In[27]:


arr1=arr
arr1


# In[28]:


arr


# In[29]:


arr1=arr.copy()
arr1


# In[30]:


print(arr)
arr1[3:]=1000
print(arr1)


# In[31]:


# some condition
val=2
arr<2


# In[33]:


arr*2


# In[34]:


arr//2


# In[35]:


arr%2


# In[36]:


arr/2


# In[37]:


arr[arr<2]


# In[38]:


arr[arr<100]


# In[41]:


arr1=np.arange(0,10).reshape(5,2)
arr1


# In[42]:


arr2=np.arange(0,10).reshape(5,2)
arr2


# In[43]:


arr1*arr2


# In[60]:


arr=np.ones((4,4),dtype=int)
arr


# In[72]:


arr = np.random.rand(3,3)
arr


# In[73]:


arr = np.random.randn(3,3)
arr


# In[71]:


import seaborn as sns
import pandas as pd


# In[61]:


sns.distplot(pd.DataFrame(arr.reshape(16,1)))


# In[64]:


np.random.randint(0,100,8).reshape(4,2)


# In[67]:


np.random.random_sample((1,5))


# In[ ]:




