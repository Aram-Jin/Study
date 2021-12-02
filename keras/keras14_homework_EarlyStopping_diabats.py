'''
Earlystopping 이란?
OverFitting 되는 경우를 방지하기 위하여 사용하는 함수
- overfitting은 training data에만 지나치게 적응되어서 그 외의 데이터에는 제대로 대응하지 못하는 상태
콜백함수란? 어떤 함수를 수행 시 그 함수에서 내가 지정한 함수를 호출하는 것.
fit() 함수에서 EarlyStopping() 콜백함수가 학습 과정 중 매번 호출된다

training dataset과 test dataset이 정확히 일치한다면, training dataset에 fitting될 수록 모델의 예측 정확도는 증가. 
그러나 문제는 training dataset과 test dataset은 조금씩 다른 경향을 보인다. 

우리의 목적: 학습을 통해 머신 러닝 모델의 underfitting된 부분을 제거하면서 overfitting이 발생하기 직전 학습을 멈추는 것인데 이를 위해 머신 러닝에서 validation set을 이용한다. 
Validation  loss가 증가하는 시점부터 overfitting이 발생했다고 판단하고, 이에 따라 학습을 중단한다. 



# restore_best_weights
Early stopping으로 일정 patience 으로 연속적인 훈련 후 값이 향상되지 않을 경우 종료시킬 것인지를 나타냄. 
이경우 restore_best_weights=True를 사용할 경우 epoch중에서 최적의 값으로 모델 복구를 도와준다. 

True라면 training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원합니다.
False라면, 마지막 training이 끝난 후의 weight로 놔둡니다

'''
# 과제 - Earlystopping되는 시점이 가장 최소의 val_loss값 지점인가? 아니요.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = load_diabetes()
x=datasets.data
y=datasets.target

print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, shuffle=True, random_state=49)

#2. 모델구성
model = Sequential() 
model.add(Dense(50, input_dim=10))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(170))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

start=time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=8, validation_split=0.5, callbacks=[es])  
end=time.time() - start

print("걸린시간:  ", round(end,3))

#4. 평가, 예측
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

'''
print("=======================================================================================")
print(hist)
print("=======================================================================================")
print(hist.history)
print("=======================================================================================")
print(hist.history['loss'])
print("=======================================================================================")
'''
print(hist.history['val_loss'])
print("=======================================================================================")


'''
20/20 [==============================] - 0s 2ms/step - loss: 3436.9167 - val_loss: 3332.5576
Epoch 00067: early stopping
걸린시간:   3.547
5/5 [==============================] - 0s 755us/step - loss: 2671.1497
loss : 2671.149658203125
r2스코어: 0.5229060663506816
[4555.07421875, 3633.919677734375, 3599.19921875, 3684.59033203125, 3511.394287109375, 4310.93310546875, 3525.454345703125, 14914.9697265625, 3541.918212890625, 3831.660400390625, 4135.82470703125, 3418.563720703125, 3947.95849609375, 3431.732177734375, 3746.34521484375, 3460.1689453125, 3202.536376953125, 3354.9404296875, 4038.93505859375, 3226.321044921875, 4221.48095703125, 3794.08544921875, 3510.547607421875, 3274.95703125, 3516.674560546875, 3589.84912109375, 5860.12109375, 5823.29345703125, 4655.99658203125, 4104.16796875, 4569.01953125, 3729.35693359375, 3241.182861328125, 3906.4072265625, 5684.80126953125, 3391.8427734375, 3664.725341796875, 3273.861083984375, 4306.51025390625, 3905.89111328125, 3550.721435546875, 3252.301513671875, 3639.9072265625, 4961.654296875, 3342.760986328125, 3237.904052734375, 5208.15185546875, 5042.42236328125, 4448.49169921875, 
4740.8759765625, 4064.265380859375, 3246.870361328125, 3329.957763671875, 3734.37060546875, 3289.13427734375, 3914.41845703125, 3577.483154296875, 3452.882568359375, 3723.6318359375, 3576.834716796875, 3615.012939453125, 4011.874267578125, 3715.58837890625, 3390.64306640625, 3354.2802734375, 3240.80322265625, 3332.5576171875]
'''
## 문제해결 과정
#1. 마지막 patience구간의 val_loss 값 중 최소값을 찾아 EarlyStopping지점의 val_loss값 과 비교하여 분석

#2. 마지막 patience구간의 val_loss값 중 최소값 : 
hist = [4555.07421875, 3633.919677734375, 3599.19921875, 3684.59033203125, 3511.394287109375, 4310.93310546875, 3525.454345703125, 14914.9697265625, 3541.918212890625, 3831.660400390625, 4135.82470703125, 3418.563720703125, 3947.95849609375, 3431.732177734375, 3746.34521484375, 3460.1689453125, 3202.536376953125, 3354.9404296875, 4038.93505859375, 3226.321044921875, 4221.48095703125, 3794.08544921875, 3510.547607421875, 3274.95703125, 3516.674560546875, 3589.84912109375, 5860.12109375, 5823.29345703125, 4655.99658203125, 4104.16796875, 4569.01953125, 3729.35693359375, 3241.182861328125, 3906.4072265625, 5684.80126953125, 3391.8427734375, 3664.725341796875, 3273.861083984375, 4306.51025390625, 3905.89111328125, 3550.721435546875, 3252.301513671875, 3639.9072265625, 4961.654296875, 3342.760986328125, 3237.904052734375, 5208.15185546875, 5042.42236328125, 4448.49169921875, 
4740.8759765625, 4064.265380859375, 3246.870361328125, 3329.957763671875, 3734.37060546875, 3289.13427734375, 3914.41845703125, 3577.483154296875, 3452.882568359375, 3723.6318359375, 3576.834716796875, 3615.012939453125, 4011.874267578125, 3715.58837890625, 3390.64306640625, 3354.2802734375, 3240.80322265625, 3332.5576171875]
print(len(hist)) # 67
print(min(hist)) # 3202.536376953125
print(hist.index(min(hist))+1) # 17

#3. EarlyStopping지점의 val_loss값 : val_loss: 3332.5576

'''
#4. [결론]
OverFitting 되는 경우를 방지하기 위하여 사용하는 함수로 EarlyStopping을 사용
훈련 데이터와는 별도로 검증 데이터(validation data)를 준비하고, 매 epoch 마다 검증 데이터에 대한 오류(validation loss)를 측정하여 모델의 훈련 종료를 제어한다. 
과적합이 발생하기 전 까지 training loss와 validaion loss 둘다 감소하지만, 과적합이 일어나면 training loss는 감소하는 반면에 validation loss는 증가한다. 
그래서 early stopping은 validation loss가 증가하는 시점에서 훈련을 멈추도록 조종한다.
위의 결과값을 보면 validaion loss가 감소하다가 17번째부턴 계속해서 증가한다고 생각할 수 있다. 
patience를 50로 설정하였기 때문에 모델의 훈련은 67번째 epoch에서 종료할 것이다. 위의 값을 보아도 EarlyStopping까지 hist의 갯수는 67개로 일치한다.

그렇다면 훈련이 종료되었을 때 이 모델의 성능은 17번째와 67번째에서 관측된 성능 중에서 어느 쪽과 일치할까? 
안타깝게도 17번째가 아닌 67번째의 성능을 지니고 있다. 위 예제에서 적용된 early stopping은 훈련을 언제 종료시킬지를 결정할 뿐이고, 
Best 성능을 갖는 모델을 저장하지는 않는다. 따라서 early stopping과 함께 모델을 저장하는 callback 함수를 반드시 활용해야만 한다.
=> "restore_best_weights" 을 사용할 수 있다.(위에 설명)


[참고]
https://deep-deep-deep.tistory.com/55
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221468928164
https://forensics.tistory.com/29

'''