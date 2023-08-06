#!/usr/bin/env python
# coding: utf-8

# # Numpy
# Numpy is the core library for scientific computing in Python. 
# It provides a highperformance multidimensional array object, and tools for working with these arrays.
# To use Numpy, we first need to import the numpy package:

# In[2]:


import numpy as np
from numpy import array


# Arrays
# A numpy array is a grid of values, all of the same type. The number of dimensions is the 
# rank of the array; the shape of an array is a tuple of integers giving the size of the array 
# along each dimension.
# We can initialize numpy arrays from nested Python lists, and access elements using square 
# brackets:
# 

# In[3]:


# 1-dim arrays
a = np.array([1, 2, 3, 4, 5, 6]) # Create a rank 1 array
print(a)


# In[4]:


print(type(a))


# In[5]:


a[0]


# # Slicing

# In[6]:


a[2 : 5]


# In[7]:


# To get the rank/axis of the array numpy provide ndim
a.ndim


# In[8]:


# Return number of values axis wise 
# RC
a.shape


# In[9]:


# Return the type of elements
a.dtype


# In[10]:


# R/C 
a.size


# In[11]:


a.itemsize #total no of bytes


# In[12]:


a


# # Multi-dim array

# In[13]:


a=[[1,2,3],
 [4,5,6], 
 [4,9,0]]
a


# In[14]:


a = np.array([[1,2,3],[4,5,6], [4,9,0]], dtype='int64') 
print(a)


# In[15]:


print(type(a))


# In[16]:


a[0]


# In[17]:


a[0][2] #####


# In[18]:


a = np.array([[1, 2 ,3] ,[4 ,5, 6,] ,[4 ,9 ,0]])
a


# In[19]:


a[0:9][0:1] #do not consider last on
a


# In[20]:


a[0:10]


# In[21]:


a


# In[22]:


#a[1:3, 1:2][ 1:3, 2:2] Invalid statement
a=array([], shape=(1, 0), dtype=int64)
a.ndim


# In[25]:


a.ndim


# In[26]:


a.shape


# In[27]:


a.size


# In[28]:


a.dtype


# In[29]:


a.itemsize


# In[30]:


a = np.array([[3,5,2,7], [9,0,1,6]])
a


# In[31]:


a.shape


# In[32]:


a_1 = np.reshape(a, (4, 2))
a_1


# In[33]:


a_1.shape


# In[34]:


# Convert into 1D
np.ravel(a)


# # Basic Operations

# In[35]:


a = np.array([5, 9, 2, -3, 8, 1], dtype='int32')
b = np.array([7, 0, 2, 5, -9, 1], dtype='int32')
a.min()


# In[36]:


np.max(a)


# In[37]:


np.min(a)


# In[38]:


np.argmax(a) #index of max value


# In[43]:


np.argmax(b)


# In[44]:


np.max(b)


# In[45]:


np.ceil(a)


# In[46]:


np.argmin(a)


# In[47]:


np.abs(a)


# In[48]:


np.absolute(a)
#The functions np.absolute () and np.abs () are essentially the same. The np.abs () function 
#is essentially a shorthand version np.absolute (). You can choose which ever one you like.


# In[52]:


np.sort(a)


# In[195]:


np.add(a, b)


# In[196]:


np.subtract(a, b)


# In[197]:


np.append(a, [99, 88])


# # Special Arrays
# Numpy also provides many functions to create special arrays:

# In[200]:


np.arange(1,8,3) #first , last and 3 spacing


# In[53]:


a = np.zeros((4,4)) # Create an array of all zeros
print(a)


# In[54]:


b = np.ones((3,5)) # Create an array of all ones
print(b)


# In[56]:


c = np.full((2,2), 'ab') # Create a constant array
print(c)


# In[57]:


d = np.eye(3) # Create a identity matrix
print(d)


# In[58]:


e = np.random.random((2,3)) # Create an array filled with random values
print(e)


# # Boolean array indexing:
# Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this 
# type of indexing is used to select the elements of an array that satisfy some condition. Here 
# is an example:

# In[59]:


a = np.array([[1,2], [3, 4], [5, 6]])
print(a)


# In[60]:


print(a)
bool_idx = (a > 2) # Find the elements of a that are bigger than 2;
                   # this returns a numpy array of Booleans of the same
                   # shape as a, where each slot of bool_idx tells
                   # whether that element of a is > 2.
print(bool_idx)


# In[61]:


a = np.array([372,4,4,5])
# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[a > 99])


# For brevity we have left out a lot of details about numpy array indexing; if you want to 
# know more you should read the documentation.

# # Math Operations
# Basic mathematical functions operate elementwise on arrays, and are available both as 
# operator overloads and as functions in the numpy module:

# In[62]:


x = np.array([[1,2],
              [3,4]], dtype=np.float64)
y = np.array([[5,6],
              [7,8]], dtype=np.float64)


# In[63]:


# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))


# In[64]:


# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))


# In[65]:


# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))


# In[66]:


# Elementwise division; both produce the array
# [[ 0.2 0.33333333]
# [ 0.42857143 0.5 ]]
print(x / y)
print(np.divide(x, y))


# In[67]:


# Elementwise square root; produces the array
# [[ 1. 1.41421356]
# [ 1.73205081 2. ]]
print(np.sqrt(x))


# In[68]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])


# In[225]:


# Inner product of vectors, multiplication and addition 
print(v.dot(w))
print(np.dot(v, w))
print(v @ w)
#You can also use the `@` operator which is equivalent to numpy's 
 #`dot` operator.


# In[228]:


# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))
print(x @ v)


# # Computation Operations

# Numpy provides many useful functions for performing computations on arrays; one of the 
# most useful is sum:
# 

# In[230]:


x = np.array([[1,2],
              [3,4]])
print(np.sum(x))                 # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))         # Compute sum of each column
print(np.sum(x, axis=1))         # Compute sum of each row; prints "[3 7]"


# In[233]:


print(x)
print("transpose\n", x.T)


# In[234]:


# Transpose 
v = np.array([[1,2,3]])
print(v )
print("transpose\n", v.T)


# In[235]:


x = np.array([65, 34, 89])
np.sin(x)


# In[236]:


np.cos(x)


# # Random Numbers

# In[237]:


x=np.random.rand(4)       # Produces random numbers 
                          # between 0 and 1
print("Random numbers \n",x)
print(type(x))


# In[239]:


a=np.random.randint(1,9,(4,4)) # Produces random int number
print(a)


# In[240]:


print(np.random.random((5,5))) # Simple and also for matrix


# In[241]:


print((np.random.randn())) # Positive and negative


# # Strings

# In[242]:


# Now string operations provided by numpy
s1="1$"
s2=" batch"
s=np.char.add(s2,s1)
print(s)


# In[243]:


print(np.char.upper(s1))


# In[244]:


print(np.char.lower(s2))


# In[245]:


print(np.char.center(s2,8,fillchar="*"))


# In[ ]:




