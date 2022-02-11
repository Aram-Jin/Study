from sklearn import datasets
from sklearn.datasets import load_wine
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
datasets = load_wine()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)   # (178, 13) (178,)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_data = ohe.fit_transform(y_data.reshape(-1,1))
print(y_data.shape)  # (178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (142, 13) (142, 3)
print(x_test.shape, y_test.shape)     # (36, 13) (36, 3)

#2.모델구성
x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 3])
w1 = tf.compat.v1.Variable(tf.random.normal([13,30]), name='weight')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.normal([1,30]), name='bias')   

Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.matmul(x, w1)+b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x, w1)+b1)

w2 = tf.compat.v1.Variable(tf.random.uniform([30,20]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([1,20]), name='bias2')

Hidden_layer2 = tf.nn.relu(tf.matmul(Hidden_layer1, w2)+b2)

w3 = tf.compat.v1.Variable(tf.random.normal([20,10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([1,10]), name='bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2, w3)+b3

w4 = tf.compat.v1.Variable(tf.random.normal([10,3]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([1,3]), name='bias4')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer3, w4)+b4)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000000001).minimize(loss)

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
    