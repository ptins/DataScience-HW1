
# coding: utf-8

# In[1]:


# imports
import numpy as np
import pandas as pd


# In[2]:


# read in data
df = pd.read_csv('./players_scores.csv').set_index('student_name')
print(df)

# ## 2 Data Visualization

# In[11]:


import matplotlib.pyplot as plt


# In[12]:


math = df['math_score']
data = df['data_science_score']


# #### 1. Q-Q Plot

# In[13]:


mathM = (math-math.min())/(math.max()-math.min())
dataM = (data-data.min())/(data.max()-data.min())

mathS = mathM.sort_values()
dataS = dataM.sort_values()

plt.figure(1)
plt.plot(mathS, dataS, 'o')
plt.xlabel('math')
plt.ylabel('data')
plt.plot([0,1], [0,1], 'k--')
plt.show()


# In[ ]:


# Since the points line above the y=x line,
# the scores are higher in the data science class,
# which means data science is the easier class.


# #### 2. Scatter Plot

# In[14]:


plt.figure(2)
plt.scatter(math, data)
fit = np.polyfit(math, data, deg=1)
plt.plot(math, fit[0]*math+fit[1], 'k--')
plt.show()


# In[15]:


# The furthest point appears to be located at
# x=76, y=87, which is associated with Joel Embiid.
# To verify, we can calculate residuals and find the maximum.
res = abs(data-(fit[0]*math+fit[1]))
print(res[res == max(res)])


# In[16]:


# We see that our guess is right - nice!
