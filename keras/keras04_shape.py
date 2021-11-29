import numpy as np

a1 = np.array([[1,2],[3,4],[5,6]])
a2 = np.array([[1,2,3],[4,5,6]])
a3 = np.array([[[1],[2],[3]],[[4],[5],[6]]])
a4 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
a5 = np.array([[[1,2,3],[4,5,6]]])
a6 = np.array([1,2,3,4,5])
a7 = np.array([[[[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]],[[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]]],[[[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]],[[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]]]])


print(a1.shape)
print(a2.shape)
print(a3.shape)
print(a4.shape)
print(a5.shape)
print(a6.shape)
print(a7.shape)
