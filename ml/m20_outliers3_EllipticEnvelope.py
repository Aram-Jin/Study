import  numpy as np
aaa = np.array([[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]])
# (2, 13) -> (13, 2)
aaa = np.transpose(aaa)   # (13, 2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.2)    # contamination을 조정해서 아웃라이어의 범위를 지정해줄수 있음

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [ 1  1  1  1  1  1  1  1  1 -1 -1  1  1]  -> 아웃라이어의 위치를 -1로 표기해줌
