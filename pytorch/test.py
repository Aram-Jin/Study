# a = [1, 2, 3]

# # print(a.__next__)

# a = [1, 2, 3]

# a_iter = iter(a)
# print(type(a_iter)) # list_iterator

# print(a_iter.__next__())
# print(a_iter.__next__())
# print(a_iter.__next__())
# print(a_iter.__next__())



# a = [1, 2, 3].__iter__()
# print(type(a))

# a = [1, 2, 3]

# a = iter(a)
# print(next(a))
# print(next(a))
# print(next(a))

# print(a.__next__()) # 1 출력
# print(a.__next__()) # 2 출력
# print(a.__next__()) # 3 출력
# print(a.__next__()) # StopIteration Exception 발생

it = iter(range(3))

print(next(it, 10))
# next(it, 10)
print(next(it, 10))
# next(it, 10)
# print(next(it, 10))
# next(it, 10)
# print(next(it, 10))
# next(it, 10)
# print(next(it, 10))