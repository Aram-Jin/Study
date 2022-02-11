from sklearn import datasets
from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)    # (506, 13) (506,)

y_data = y_data.reshape(506,1)
print(y_data.shape)   # (506, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (354, 13) (354, 1)
print(x_test.shape, y_test.shape)     # (152, 13) (152, 1)

#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])
w1 = tf.compat.v1.Variable(tf.random.normal([13,6]), name='weight1')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.normal([6]), name='bias1')   

Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x, w1)+b1)

w2 = tf.compat.v1.Variable(tf.random.normal([6,11]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.normal([11]), name='bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1, w2)+b2)

w3 = tf.compat.v1.Variable(tf.random.normal([11,24]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([24]), name='bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2, w3)+b3

w4 = tf.compat.v1.Variable(tf.random.normal([24,50]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([50]), name='bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3, w4)+b4

w5 = tf.compat.v1.Variable(tf.random.normal([50,80]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([80]), name='bias5')

Hidden_layer5 = tf.matmul(Hidden_layer4, w5)+b5

w6 = tf.compat.v1.Variable(tf.random.normal([80,100]), name='weight6')
b6 = tf.compat.v1.Variable(tf.random.normal([100]), name='bias6')

Hidden_layer6 = tf.matmul(Hidden_layer5, w6)+b6

w7 = tf.compat.v1.Variable(tf.random.normal([100,80]), name='weight7')
b7 = tf.compat.v1.Variable(tf.random.normal([80]), name='bias7')

Hidden_layer7 = tf.matmul(Hidden_layer6, w7)+b7

w8 = tf.compat.v1.Variable(tf.random.normal([80,50]), name='weight8')
b8 = tf.compat.v1.Variable(tf.random.normal([50]), name='bias8')

Hidden_layer8 = tf.matmul(Hidden_layer7, w8)+b8

w9 = tf.compat.v1.Variable(tf.random.normal([50,25]), name='weight9')
b9 = tf.compat.v1.Variable(tf.random.normal([25]), name='bias9')

Hidden_layer9 = tf.matmul(Hidden_layer8, w9)+b9

w10 = tf.compat.v1.Variable(tf.random.normal([25,12]), name='weight10')
b10 = tf.compat.v1.Variable(tf.random.normal([12]), name='bias10')

Hidden_layer10 = tf.matmul(Hidden_layer9, w10)+b10

w11 = tf.compat.v1.Variable(tf.random.normal([12,5]), name='weight11')
b11 = tf.compat.v1.Variable(tf.random.normal([5]), name='bias11')

Hidden_layer11 = tf.matmul(Hidden_layer10, w11)+b11

w12 = tf.compat.v1.Variable(tf.random.normal([5,2]), name='weight12')
b12 = tf.compat.v1.Variable(tf.random.normal([2]), name='bias12')

Hidden_layer12 = tf.matmul(Hidden_layer11, w12)+b12

w13 = tf.compat.v1.Variable(tf.random.normal([2,1]), name='weight13')
b13 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias13')

hypothesis = tf.matmul(Hidden_layer12, w13)+b13

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000000000000001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    _, loss_val, w_val= sess.run([train, loss, w13], feed_dict={x:x_train, y:y_train})
    print(epochs, '\t', loss_val, '\t', w_val)
           
#4. 예측
predict = tf.matmul(Hidden_layer12, w_val) + b13   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x:x_test, y:y_test})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)
