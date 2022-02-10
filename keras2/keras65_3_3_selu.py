import numpy as np
import matplotlib.pyplot as plt

def selu(x):
    return np.where (x <= 0, scale * alpha * (np.exp(x)-1), scale * x)
         
x = np.arange(-5, 5, 0.1)
alpha = 1
scale = 1
y = selu(x, alpha, scale)

plt.plot(x, y)
plt.grid()
plt.show()