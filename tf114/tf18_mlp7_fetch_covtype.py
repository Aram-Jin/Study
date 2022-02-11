from sklearn import datasets
from sklearn.datasets import fetch_covtype
import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = fetch_covtype()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)   # (581012, 54) (581012,)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_data = ohe.fit_transform(y_data.reshape(-1,1))
print(y_data.shape)   # (581012, 7)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (464809, 54) (464809, 7)
print(x_test.shape, y_test.shape)     # (116203, 54) (116203, 7)

#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 54])
y = tf.placeholder(tf.float32, shape=[None, 7])
w1 = tf.compat.v1.Variable(tf.random.normal([54,50]), name='weight1')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.normal([1,50]), name='bias1')   

Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x, w1)+b1)

w2 = tf.compat.v1.Variable(tf.random.normal([50,20]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.normal([1,20]), name='bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1, w2)+b2)

w3 = tf.compat.v1.Variable(tf.random.normal([20,10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([1,10]), name='bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2, w3)+b3

w4 = tf.compat.v1.Variable(tf.random.normal([10,7]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([1,7]), name='bias4')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer3, w4)+b4)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.004).minimize(loss)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer,loss], feed_dict={x:x_train, y:y_train})
        if step % 200 ==0:
            print(step, loss_val)
    
    results = sess.run(hypothesis, feed_dict={x:x_test})
    print(results, sess.run(tf.math.argmax(results, 1)))     #[[9.3190324e-01 6.8059169e-02 3.7637248e-05]] [0]
    
    accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(y_test, results), dtype=tf.float32)))
    pred = sess.run(tf.math.argmax(results, 1), feed_dict={x:x_test, y:y_test})
    
    print("예측결과 : ", pred)
    print("accuracy : ", accuracy)

    sess.close()
    
    
# 예측결과 :  [0 0 0 ... 0 0 0]
# accuracy :  0.0