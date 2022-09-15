import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)

x_train = np.arange(0, 50, 1).reshape(50,1)
noise=300*np.random.rand(50, 1)
x_train = x_train.reshape(-1, 1)
y_train = 1 + np.square(x_train) + noise
y_train = np.ravel(y_train)

x_features = ['x1','x2']
scaler = StandardScaler()
X_norm = scaler.fit_transform(x_train)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b.
y_pred = np.dot(X_norm, w_norm) + b_norm
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")
print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

x = np.linspace(-2,2,100)
y = w_norm*x+b_norm
plt.plot(x, y, '-r', label=f"{w_norm}x+{b_norm}")
plt.plot(X_norm, y_train, '*')
plt.show()
