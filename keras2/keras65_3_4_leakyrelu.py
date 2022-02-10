import numpy as np
import matplotlib.pyplot as plt

def leakyrelu(x):
    return (x>=0)*x + (x<0)*0.01*x # 알파값(보통 0.01) 조정가능
    # return np.maximum(0.01*x,x)
     
x = np.arange(-5, 5, 0.1)
y = leakyrelu(x)

plt.plot(x, y)
plt.grid()
plt.show()