from matplotlib.pyplot import axis
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
(x_train, _ ), (x_test, _ ) = mnist.load_data()

print(x_train.shape, x_test.shape)   # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
# print(x.shape)   # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)   # (70000, 784)

########################################################
# pca를 통해서 0.95 이상인 n_components 가 몇개???
# 0.95
# 0.00
# 0.999
# 1.0
# np.argmax 쓰기
########################################################

pca = PCA(n_components=784)
x = pca.fit_transform(x)
# print(x)
# print(x.shape)  

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.95) + 1)  # 154
print(np.argmax(cumsum >= 0.99) + 1)  # 331
print(np.argmax(cumsum >= 0.999) + 1)  # 486
print(np.argmax(cumsum == 1.0) + 1)  # 1

print(np.argmax(cumsum) +1)  # 713


# result_val = [0.95,0.99,0.999,1.0]
# for i in result_val:
#     print(i,np.argmax(cumsum>i))
    
# '''
# 0.95 153
# 0.99 330
# 0.999 485
# 1.0 712
# '''

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# # plt.plot(pca_EVR)
# plt.grid()
# plt.show()
