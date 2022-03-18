import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.uniform(high=1, size=100).reshape(-1,1)
y = np.sin(x+.25 * np.pi).reshape(-1)

model = LinearRegression()
model.fit(x, y)

x_pred = np.linspace(0,1, 100).reshape(-1,1)
y_pred = model.predict(x_pred)

plt.scatter(x,y, label='data')
plt.plot(x_pred, y_pred,  label="model")

plt.legend()
plt.savefig('./plot.png')