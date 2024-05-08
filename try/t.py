#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt


ord = 5
a = np.arange(6)
y = np.array([85.0, 79.19, 74.75, 71.19, 68.19, 65.10]).astype(np.float32)
x = np.stack([a**i for i in range(ord + 1)], axis=1)
print(x)

theta = np.linalg.inv(x.T.dot(x)).dot(x.T)
theta = theta.dot(y)
print(f"{np.round(theta, 3)}")


t = np.stack([3**i for i in range(ord + 1)], axis=0)
y = theta.dot(t)
print(y)


x = np.arange(30).astype(np.float32)
t = np.stack([x**i for i in range(ord + 1)], axis=0)
y_ = theta.dot(t)
plt.plot(x, y_)
plt.scatter(np.arange(6), np.array([85.0, 79.19, 74.75, 71.19, 68.19, 65.10]))
plt.savefig("a.png")
