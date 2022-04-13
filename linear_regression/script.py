import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#  A system of linear equations consists of multiple, related functions with a common set of variables. The word linear equation is often used interchangeably with linear function. Many real world processes can be modeled using multiple, related linear equations.
# Transform x using the equation 
#  y=30x+1000  and assign the result to y1.
# Transform x using the equation 
# y=50x+100  and assign the result to y2

x = np.linspace(0, 50, 1000)
y1 = 30 * x + 1000
y2 = 50 * x + 100

plt.plot(x, y1, c='blue')
plt.plot(x, y2, c='red')
# both functions intersect at somewhere near the point 
# (45,2200), this is a solution.

#  Gaussian elimination: using linear algebra

data = pd.read_csv('AmesHousing.txt', delimiter='\t')
train = data[0:1460]
test = data[1460:]
print(train.info())
target = "SalePrice"
 

fig = plt.figure(figsize=(7, 15))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)                    
train.plot(x='Garage Area', y='SalePrice', ax=ax1, kind='scatter')
train.plot(x='Gr Liv Area', y='SalePrice', ax=ax2, kind='scatter')
train.plot(x='Overall Cond', y='SalePrice', ax=ax3, kind='scatter')
plt.show()
# correlation between pairs of these columns using the pandas.DataFrame.corr() method:

print(train[['Garage Area', 'Gr Liv Area', 'Overall Cond', 'SalePrice']].corr())