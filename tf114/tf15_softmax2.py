from unittest import result
import numpy as np
from sklearn.utils import resample
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]     # (8, 4)
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]       # (8, 3)

x_predict = [[1,11,7,9]]   # (1,4) -> (N,4)

#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])
w = tf.compat.v1.Variable(tf.random.normal([4,3]), name = 'weight1')   # (4,3)
b = tf.compat.v1.Variable(tf.random.normal([1,3]), name = 'bias')

# hypothesis = x * w + b
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax'))

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04).minimize(loss)


#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer,loss], feed_dict={x:x_data, y:y_data})
        if step % 200 ==0:
            print(step, loss_val)
    
    results = sess.run(hypothesis, feed_dict={x:x_data})
    print(results, sess.run(tf.math.argmax(results, 1)))     #[[9.3190324e-01 6.8059169e-02 3.7637248e-05]] [0]
   
    y_predict = sess.run(hypothesis, feed_dict={x:x_predict})
    print("예측 : ", y_predict, sess.run(tf.math.argmax(y_predict, 1)))
    

    accuracy = tf.reduce_mean(tf.cast(tf.equal(results, y_predict), dtype=tf.float32))
    pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_predict})


    print("예측결과 : ", pred)
    print("accuracy : ", acc)

    sess.close()
