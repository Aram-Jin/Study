from sklearn import datasets
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)    # (569, 30) (569,)

y_data = y_data.reshape(569,1)
print(y_data.shape)   # (569, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (455, 30) (455, 1)
print(x_test.shape, y_test.shape)     # (114, 30) (114, 1)

#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])
w1 = tf.compat.v1.Variable(tf.random.normal([30,50]), name='weight')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.normal([50]), name='bias')   

Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x, w1)+b1)

w2 = tf.compat.v1.Variable(tf.random.uniform([50,10]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([10]), name='bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1, w2)+b2)

w3 = tf.compat.v1.Variable(tf.random.normal([10,10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2, w3)+b3

w4 = tf.compat.v1.Variable(tf.random.normal([10,1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias4')

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer3, w4)+b4)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(101):
    loss_val, hypothesis_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})  
    if epoch % 200 == 0:
        print(epoch, 'loss : ', loss_val, '\n', hypothesis_val)
               
#4. 평가, 예측
y_predict = tf.cast(hypothesis > 0.5, dtype=tf.int32)  
# print(y_predict)   # Tensor("Cast:0", shape=(?, 1), dtype=float32)
# print(sess.run(hypothesis > 0.5, feed_dict={x:x_data, y:y_data}))

accuracy = tf.reduce_mean(tf.cast(tf.equal(y_data, y_predict), dtype=tf.float32))

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print("===============================================================================")
print("예측값 : \n", hypothesis_val)
print("예측결과 : ", pred)
print("accuracy : ", acc)

sess.close()

# accuracy :  0.37258348