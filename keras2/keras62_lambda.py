# lambda 함수

gradient = lambda x: 2*x - 4   # 여기서 gradient는 함수명, lambda 함수를 이용해 한줄로 간단하게 사용할 수 있음

def gradient2(x):
    temp = 2*x -4
    return temp

x = 3

print(gradient(x))   # 2
print(gradient2(x))  # 2

