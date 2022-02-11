from tensorflow.keras.datasets import mnist
from sklearn import datasets
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
print(y_train.shape) # (60000, 10)
y_test = ohe.fit_transform(y_test.reshape(-1,1))
print(y_test.shape) # (10000, 10)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])  
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]) 

print(x_train.shape, y_train.shape)  # (60000, 784) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 784) (10000, 10)

#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
w1 = tf.compat.v1.Variable(tf.random.normal([784,128]), name='weight1')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.normal([1,128]), name='bias1')   

Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x, w1)+b1)
layers = tf.nn.dropout(Hidden_layer1, keep_prob=0.8)

w2 = tf.compat.v1.Variable(tf.random.normal([128,64]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.normal([1,64]), name='bias2')

Hidden_layer2 = tf.nn.relu(tf.matmul(Hidden_layer1, w2)+b2)
layers = tf.nn.dropout(Hidden_layer2, keep_prob=0.8)

w3 = tf.compat.v1.Variable(tf.random.normal([64,32]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([1,32]), name='bias3')

Hidden_layer3 = tf.nn.relu(tf.matmul(Hidden_layer2, w3)+b3)
layers = tf.nn.dropout(Hidden_layer3, keep_prob=0.8)

w4 = tf.compat.v1.Variable(tf.random.normal([32,16]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([1,16]), name='bias4')

Hidden_layer4 = tf.nn.relu(tf.matmul(Hidden_layer3, w4)+b4)
layers = tf.nn.dropout(Hidden_layer4, keep_prob=0.8)

w5 = tf.compat.v1.Variable(tf.random.normal([16,8]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([1,8]), name='bias5')

Hidden_layer5 = tf.nn.relu(tf.matmul(Hidden_layer4, w5)+b5)
layers = tf.nn.dropout(Hidden_layer5, keep_prob=0.8)

w6 = tf.compat.v1.Variable(tf.random.normal([8,10]), name='weight6')
b6 = tf.compat.v1.Variable(tf.random.normal([1,10]), name='bias6')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer5, w6)+b6)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000001).minimize(loss)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer,loss], feed_dict={x:x_train, y:y_train})
        if step % 200 ==0:
            print(step, loss_val)
    
    results = sess.run(hypothesis, feed_dict={x:x_test})
    print(results, sess.run(tf.math.argmax(results, 1)))     #[[9.3190324e-01 6.8059169e-02 3.7637248e-05]] [0]
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, results), dtype=tf.float32))
    pred, acc = sess.run([tf.math.argmax(results, 1), accuracy], feed_dict={x:x_test, y:y_test})

    print("예측결과 : ", pred)
    print("accuracy : ", acc)

    sess.close()