import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

data = np.array(range(1, 101))
x_predict = np.array(range(96, 106))

# size = 5 로 나누자!   x 4개, y 1개

def split_x(dataset, size):
    list = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        list.append(subset)
    return np.array(list)

train = split_x(data, 5)
#print(train)
#print(train.shape)     # (96, 5)

x = train[:, :4]       # :은 모든 행또는 열, 마지막은 -1로쓸수있음. 여기선 [:, :4] 과 [:, :-1] 똑같음
y = train[:, 4]
#print(x,y)
#print(x.shape, y.shape)   # (96, 4) (96,)

x = x.reshape(96, 4, 1)

predict = split_x(x_predict, 4)
#print(predict)  
#print(predict.shape)   # (7, 4)

predict = predict.reshape(7, 4, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(4,1), return_sequences=True ))       # (N,4,1) -> (N,64) activation='relu'
model.add(GRU(16, return_sequences=False, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')      # optimizer는 loss를 최적화한다
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
model.evaluate(x, y)

result = model.predict(predict)
print(result)


'''
[[ 99.35497 ]
 [100.00301 ]
 [100.62409 ]
 [101.21872 ]
 [101.78753 ]
 [102.331215]
 [102.85055 ]]
 '''


