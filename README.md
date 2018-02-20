# CSE40647/CSE60647 - HW1

This is the first assignment for my Data Science Class.

## Getting Started

This project uses Python 3.

Clone this repo with the following command.

```
git clone https://github.com/ptins/HW1-DataScience.git
```

## Set Up

```
# imports
import numpy as np
import pandas as pd

# read in data
df = pd.read_csv('./players_scores.csv').set_index('student_name')
print(df)
```

## 1 Data Description

```
scores = df['data_science_score']
# scores = df['math_score']
```

Calculate mean, median, and mode of Data Science scores.

```
print('Mean: {}, Median: {}, Mode: {}'.format(
scores.mean(), scores.median(), scores.mode().values))
```

Calculate variance and standard deviation of Data Science scores.

```
print('Variance: {}, Standard Deviation: {}'.format(
scores.std()**2, scores.std()))
```

Define functions to calculate the online mean and online variance.

```
def incremental_mean(mu, n, x_new):
    return ((n*mu)+x_new)/(n+1)
```
```
def incremental_var(v, mu, n, x_new):
    mu_new = incremental_mean(mu, n, x_new)
    return ((n-1)*(v) + (x_new - mu)*(x_new - mu_new))/n
```

## 2 Data Visualization

Q-Q plot. The X-axis is Math score. The Y-axis is Data Science score. Add a proper
dashed line to answer the question: Which course is easier for the students, Math or
Data Science?

```
import matplotlib.pyplot as plt
math = df['math_score']
data = df['data_science_score']
```
```
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
```

Since the points line above the y=x line, 
the scores are higher in the data science class, 
which means data science is the easier class.


Scatter plot. The X-axis is Math score. The Y-axis is Data Science score. Draw a linear
regression dashed line to answer the question: Which student is more likely to be an
outlier (farthest from the line)?

```
plt.figure(2)
plt.scatter(math, data)
fit = np.polyfit(math, data, deg=1)
plt.plot(math, fit[0]*math+fit[1], 'k--')
plt.show()
```

The furthest point appears to be located at x=76, y=87, which is 
associated with Joel Embiid. To verify, we can calculate residuals and find the maximum.

```
res = abs(data-(fit[0]*math+fit[1]))
print(res[res == max(res)])
```

## 3 Data Reduction

Use Python to call a Singular Value Decomposition (SVD) package and calculate left
singular vector U (size: n × k) and singular values λi (i = 1 . . . k) where the number of
singular values k is set as 2. The goal is to reduce the number of features from n to k.

```
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
```

Use the following command to read in the adjacency matrix, and cast the type to float.
```
A = pd.read_csv('./social_graph.csv').set_index('name').as_matrix().astype(float)
```
Call the scipy.sparse.linalg function on the function with k=2 to invoke Singular Value Decomposition.
```
U0, S, Vt = linalg.svds(A, k=2)
print('U0:\n{}\n\nS:\n{}\n\nVt:\n{}'.format(U0, S, Vt))
```
To better understand the clusters, use the following commands to plot the reduced data.
```
plt.figure(3)
pdscatter = plt.scatter(U0[:,0],U0[:,1])
plt.title('SVD for Adjacency Matrix')
plt.show()
```
We see that there exist two clusters, one with 6 students, and the other with 3.

