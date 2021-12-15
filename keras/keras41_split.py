import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)     # (6, 5) 

x = bbb[:, :4]      # 파이썬=> :은 모든 행또는 열, 마지막은 -1로쓸수있음. 여기선 [:, :4] 과 [:, :-1] 똑같음
y = bbb[:, 4]
print(x,y)
print(x.shape, y.shape)   # (6, 4) (6,)