import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.random.uniform(high=1, size=100).reshape(-1,1)
y = np.sin(x+.25 * np.pi).reshape(-1)

poly = PolynomialFeatures(degree = 2 )
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

x_pred = np.linspace(0,1, 100).reshape(-1,1)
x_poly_pred = poly.transform(x_pred)
y_pred = model.predict(x_poly_pred)

plt.scatter(x,y, label='data')
plt.plot(x_pred, y_pred,  label="model")

plt.legend()
plt.savefig('./plot.png')
