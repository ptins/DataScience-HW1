# imports
import numpy as np
import pandas as pd

# ## 3 Data Reduction


# In[17]:


import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt


# In[18]:


A = pd.read_csv('./social_graph.csv').set_index('name').as_matrix().astype(float)


# In[19]:


U0, S, Vt = linalg.svds(A, k=2)
print('U0:\n{}\n\nS:\n{}\n\nVt:\n{}'.format(U0, S, Vt))


# In[20]:


print('{}\n{}\n{}'.format(U0.shape, S.shape, Vt.shape))


# In[24]:


plt.figure(3)
pdscatter = plt.scatter(U0[:,0],U0[:,1])
plt.title('SVD for Adjacency Matrix')
plt.show()


# In[22]:


U0


# In[ ]:


# It seems that one cluster of students is A-B-C-D-G-H
# The second cluster consists of E-F-I.
