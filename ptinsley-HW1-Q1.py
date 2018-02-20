
# coding: utf-8

# In[1]:


# imports
import numpy as np
import pandas as pd


# In[2]:


# read in data
df = pd.read_csv('./players_scores.csv').set_index('student_name')
print(df)


# ## 1 Data Description

# In[3]:


scores = df['data_science_score']
# scores = df['math_score']


# #### 1. Calculate mean, median, and mode of Data Science scores

# In[4]:


print('Mean: {}, Median: {}, Mode: {}'.format(
scores.mean(), scores.median(), scores.mode().values))


# #### 2. Calculate variance and standard deviation of Data Science scores

# In[5]:


print('Variance: {}, Standard Deviation: {}'.format(
scores.std()**2, scores.std()))


# #### 3. Incremental Mean/Variance Functions

# In[6]:


def incremental_mean(mu, n, x_new):
    return ((n*mu)+x_new)/(n+1)


# In[7]:


def incremental_var(v, mu, n, x_new):
    mu_new = incremental_mean(mu, n, x_new)
    return ((n-1)*(v) + (x_new - mu)*(x_new - mu_new))/n


# In[8]:


u_prime = incremental_mean(mu=scores.mean(), n=len(scores), x_new=100)
print('u\' = {}'.format(u_prime))


# In[9]:


v_prime = incremental_var(scores.std()**2, scores.mean(), len(scores), 100)
print('v\' = {}'.format(v_prime))


# #### Verify Function Correctness

# In[10]:


scores_plus = scores.append(pd.Series(100))
print('u\' = {}'.format(scores_plus.mean()))
print('v\' = {}'.format(scores_plus.std()**2))
