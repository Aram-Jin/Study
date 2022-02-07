import autokeras as ak
import tensorflow as tf
# from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.cifar100.load_data()
    # tf.keras.datasets.mnist.load_data()

#2. 모델
model = ak.ImageClassifier(overwrite=True,
                           max_trials=5    # 반복횟수
                           )

#3. 컴파일, 훈련
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)   # [2.5616681575775146, 0.3540000021457672]

model.summary()