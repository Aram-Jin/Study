import numpy as np 
            

x = np.array([[1,2,3,4,5,6,7,8,9,10], [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

# x = x.reshape(10,2) -> 순서가 바뀌지 않음
# x = np.transpose(x)   
# x = x.T 

# print(x.shape) #(2,10)
# x = np.transpose(x)
# print(x.shape)
# print(x)

print(y)
a = y.reshape(-1,1)
b = a.reshape(1,-1)
print(a)
print(b)
