import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# The cost function
def cost_fun(x, y, w, b):
    cost_sum = 0
    m = x.shape[0]
    f_wb=np.dot(w,x) + b
    cost = (f_wb - y) ** 2
    cost_sum = np.sum(cost)
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

#Initiallize w and b
w = 10*np.ones(2)
b = 100
#Train values 
x_train = np.array([[1.0, 1.7, 1.9, 2.0, 2.3, 2.5, 2.1, 2.18,  2.7, 3.0, 3.2, 3.5],[0.5, 2.4, 5, 2, 10, 8, 13, 5,  2.7, 4.7, 8.2, 3.9]])
y_train = np.array([250, 300, 400, 480, 460, 430, 440, 410,  600, 630, 730, 700])
m = y_train.shape[0] #number of samples(the size of y)
#Not important!
n = x_train.shape[0]
dw=np.ones(n)
db=1
condition = 1
# Learning factor
a=0.00001
iteration=0
cost_save = []
while (condition> 0.1):# The stopping criteria is when w and b does not change significantly in two subsequent iterations.
    iteration+=1
    dj_w = np.zeros(n)
    dj_b = 0
    for j in range(n):
        #print(f"j={j}")
        for i in range(m):
            # Gradient descent method
            #dj_w and dj_b are derivatives of j wrt w and b, respectively.
            dj_w[j]= dj_w[j]+ ((w[j] * x_train[[j],[i]] + b - y_train[i]) * x_train[[j],[i]])/ m
            dj_b= dj_b+ (w[j] * x_train[[j],[i]] + b - y_train[i]) / m
    w_new = w - a * dj_w
    b_new = b - a * dj_b
    dw = w_new - w
    #print(f"dw = {dw}")
    db= b_new - b
    w = w_new
    b = b_new
    cost_save.append( cost_fun(x_train, y_train, w, b))
    if iteration > 1:
        condition = abs (cost_save[-1] - cost_save[-2])
    #print(f"condition = {condition}" )
    
cost = cost_fun(x_train, y_train, w, b)
(x, y) = np.meshgrid(np.arange(0.2, 7,1), np.arange(0.2, 7, 1))
z = w[0]*x+ w[1]*y + b

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
# Plot a 3D surface
ax.plot_surface(x, y, z)
plt.xlabel('x1')
plt.ylabel('x2')
#plt.show()
print(f"Number of iterations is {iteration}." )
print(f"The optimal w-value is {w}, and the optimal b-value is {b}.")
print(f"The value of cost function for the obtained values of w and b is {cost}")
ax.plot3D(x_train[0], x_train[1], y_train, 'r*')
plt.show()
plt.figure()
plt.plot(range(iteration),cost_save, '*')
plt.show()