import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.randint(5, size=30).reshape((-1, 1))
y = np.random.randint(2, size=30)
model = LinearRegression()
model.fit(x,y)
r = model.score(x,y)
print(r)
plt.plot(x, y)
plt.show()