import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = np.loadtxt("./housing.data")
# print(data.shape)

reg = linear_model.LinearRegression(normalize=True)

x = data[:, :13]
y = data[:, 13]

reg.fit(x, y)

print(reg.coef_)
print(mean_absolute_error(y, reg.predict(x)))
print(mean_squared_error(y, reg.predict(x)))
