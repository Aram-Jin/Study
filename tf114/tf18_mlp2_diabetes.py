from sklearn import datasets
from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)    #(442, 10) (442,)

y_data = y_data.reshape(442,1)
print(y_data.shape)   # (442, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (354, 13) (354, 1)
print(x_test.shape, y_test.shape)     # (152, 13) (152, 1)

#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])
w1 = tf.compat.v1.Variable(tf.random.uniform([10,16]), name='weight1')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.uniform([16]), name='bias1')   

Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x, w1)+b1)

w2 = tf.compat.v1.Variable(tf.random.uniform([16,12]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([12]), name='bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1, w2)+b2)

w3 = tf.compat.v1.Variable(tf.random.normal([12,8]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([8]), name='bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2, w3)+b3

w4 = tf.compat.v1.Variable(tf.random.normal([8,6]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([6]), name='bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3, w4)+b4

w5 = tf.compat.v1.Variable(tf.random.normal([6,1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias5')

hypothesis = tf.matmul(Hidden_layer4, w5)+b5

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(31):
    _, loss_val, w_val= sess.run([train, loss, w5], feed_dict={x:x_train, y:y_train})
    print(epochs, '\t', loss_val, '\t', w_val)
           
#4. 예측
predict = tf.matmul(Hidden_layer4, w_val) + b5   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x:x_test, y:y_test})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)

# r2스코어 :  0.11109720943806856
# mae :  64.06360261895684
