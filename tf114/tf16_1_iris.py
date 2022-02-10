from sklearn import datasets
from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)   # (150, 4) (150,)

y_data = y_data.reshape(150,1)
print(y_data.shape)    # (150, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (120, 4) (120, 1)
print(x_test.shape, y_test.shape)     # (30, 4) (30, 1)

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.compat.v1.Variable(tf.random.normal([4,1]), name='weight')    # y = x * w  
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')   

# hypothesis = x * w + b
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax'))

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
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, results), dtype=tf.float32))
    pred, acc = sess.run([tf.math.argmax(results, 1), accuracy], feed_dict={x:x_data, y:y_data})

    print("예측결과 : ", pred)
    print("accuracy : ", acc)

    sess.close()

# 예측결과 :  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# accuracy :  0.33333334