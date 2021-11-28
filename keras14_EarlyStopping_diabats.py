from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time
from sklearn import datasets
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
# datasets = load_boston()
datasets = load_diabetes()
x = datasets.data
y = datasets.target
'''
print(x)  
print(y)
print(x.shape)  # (442, 10) (442,)
print(y.shape)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=1, 
                 validation_split=0.2, callbacks=[es])
end = time.time() - start

print("걸린시간: ", round(end, 3),'초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
print("======================================")
print(hist)
print("======================================")
print(hist.history)
print("======================================")
print(hist.history['loss'])
print("======================================")
print(hist.history['val_loss'])

'''
'''
Epoch 111/500
282/282 [==============================] - 0s 659us/step - loss: 2977.8667 - val_loss: 3105.9443
Epoch 00111: early stopping
걸린시간:  20.79 초
3/3 [==============================] - 0s 1ms/step - loss: 3279.8750
loss : 3279.875
r2스코어 :  0.49462947754015474

[4044.382080078125, 3178.931884765625, 3163.958740234375, 5235.953125, 3298.36328125, 3736.89794921875, 3070.9921875, 3395.771240234375, 3365.360107421875, 3164.052001953125, 3241.8125, 3359.1103515625, 3339.43212890625, 3090.019287109375, 3134.33447265625, 3395.394775390625, 3063.783203125, 3078.44189453125, 4331.76513671875, 4139.15625, 3577.302734375, 4131.78564453125, 3880.5703125, 3190.9580078125, 4124.10302734375, 3268.191162109375, 3265.970947265625, 3104.85595703125, 3141.092041015625, 3160.38427734375, 3224.85400390625, 3261.8955078125, 3215.546142578125, 3648.3046875, 3176.92333984375, 3608.57470703125, 3700.447509765625, 3152.021728515625, 3304.36279296875, 3088.633544921875, 3315.700927734375, 3320.181884765625, 3052.327880859375, 3145.087158203125, 3197.845703125, 3889.761474609375, 3138.063720703125, 3146.663330078125, 3566.853759765625, 3344.53271484375, 3069.166259765625, 3061.494140625, 3107.669921875, 3066.760009765625, 3695.072265625, 3112.804931640625, 3087.8056640625, 3092.278076171875, 3159.93701171875, 3600.25732421875, 3020.70751953125, 3246.40087890625, 3150.471435546875, 3125.052001953125, 3162.94580078125, 3116.861083984375, 3180.854248046875, 3116.70263671875, 3287.52099609375, 3329.7861328125, 3078.33740234375, 3047.9052734375, 3208.109375, 3068.043212890625, 3152.186279296875, 3108.5556640625, 3428.759033203125, 4811.0, 3044.14306640625, 3126.79833984375, 3254.4482421875, 3089.5673828125, 3272.796875, 3122.289794921875, 3201.3427734375, 3375.803466796875, 3170.8525390625, 3183.74169921875, 3212.88720703125, 3688.224365234375, 3116.031005859375, 3119.1962890625, 3166.42041015625, 3202.92724609375, 3082.151123046875, 3335.23486328125, 3315.68798828125, 3132.8916015625, 3135.844482421875, 3057.6962890625, 3142.069580078125, 3768.90283203125, 3109.068603515625, 3059.559326171875, 3141.621337890625, 3089.970947265625, 3144.208740234375, 3648.30859375, 3416.248779296875, 3216.791259765625, 3105.9443359375]
'''




'''
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
'''
