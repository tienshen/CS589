# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Load data
X = np.load('../../Data/X_train.npy')
Y = np.load('../../Data/y_train.npy')
#%% Plotting mean of the whole dataset
x_mean = X.mean(axis = 0)

print(X.shape)
plt.imshow(x_mean.reshape((28,28)))
plt.title("Mean of the Dataset")
# plt.show()
fig=plt.figure()
#%% Plotting each digit
digits = [[] for i in range(10)]
for i in range(X.shape[0]):
    digits[Y[i]].append(X[i])

pic = [0 for x in range(10)]
for i in range(len(digits)):
    pic[i] = np.mean(digits[i], axis = 0).reshape((28,28))
    fig.add_subplot(5,2,i+1)
    plt.imshow(pic[i])


#%% Center the data (subtract the mean)
X_center = X - x_mean

#%% Calculate Covariate Matrix
cov_mat = np.cov(X_center.T)
# print(cov_mat.shape)

#%% Calculate eigen values and vectors
values, vec = np.linalg.eig(cov_mat)
values = values.astype(np.float)
vec = vec.T.astype(np.float)
# print(eig_vals[0].shape)

#%% Plot eigen values
plt.figure(3)
plt.plot(values[:50])
plt.title("Eigen Values")
#%% Plot 5 first eigen vectors
fig = plt.figure(4)
plt.title("5 first eigen vectors")
# sorted_index = eig_vals[1].argsort(axis = 0)
vec = vec[np.argsort(-values)]

for i in range(5):
    fig.add_subplot(1, 5, i + 1)
    plt.imshow(vec[i].reshape((28,28)))

#%% Project to two first bases
fig = plt.figure(5)
ax = fig.add_subplot(111)
digits = np.array(digits)
for i in range(10):
    x = np.matmul(digits[i], (vec[0]))
    y = np.matmul(digits[i], (vec[1]))
    ax.scatter(x, y, label=i)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#%% Plotting the projected data as scatter plot

plt.show()
