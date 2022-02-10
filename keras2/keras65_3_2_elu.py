import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)
     
x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x, y)
plt.grid()
plt.show()