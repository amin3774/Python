import numpy as np
import matplotlib.pyplot as plt

# The cost function
def cost_fun(x, y, w, b):
    cost_sum = 0
    m = x.shape[0]
    f_wb = w * x + b
    cost = (f_wb - y) ** 2
    cost_sum = np.sum(cost)
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

#Initiallize w and b
w = 0
b = 0
#Train values 
x_train = np.array([1.0, 1.7, 1.9, 2.0, 2.3, 2.5, 2.1, 2.18,  2.7, 3.0, 3.2, 3.5])
y_train = np.array([250, 300, 400, 480, 460, 430, 440, 410,  600, 630, 730, 700])
m = x_train.shape[0]
#Not important!
dw=1
db=1
# Learning factor
a=0.1
iteration=0
while (dw > 0.01) & (db > 0.01):# The stopping criteria is when w and b does not change significantly in two subsequent iterations.
    iteration+=1
    dj_w = 0
    dj_b = 0
    for i in range(m):
        # Gradient descent method
        #dj_w and dj_b are derivatives of j wrt w and b, respectively.
        dj_w= dj_w+ ((w * x_train[i] + b - y_train[i]) * x_train[i])/ m
        dj_b= dj_b+ (w * x_train[i] + b - y_train[i]) / m
    w_new = w - a * dj_w
    b_new = b - a * dj_b
    dw = w_new - w
    db= b_new - b
    w = w_new
    b = b_new
cost = cost_fun(x_train, y_train, w, b)
x = np.linspace(0,4,100)
y = w*x+b
plt.plot(x, y, '-r', label=f"{w}x+{b}")
print(f"Number of iterations is {iteration}." )
print(f"The optimal w-value is {w}, and the optimal b-value is {b}.")
print(f"The value of cost function for the obtained values of w and b is {cost}")

plt.plot(x_train, y_train, '*')
plt.show()