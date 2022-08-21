a = (1, 2)
b = [3, 4]

c = a + tuple(b)
print(c)

a = (1, 2, 3)
#a.append(4) #리스트 전용
a = a + (4,) # a = 4,
print(a)
