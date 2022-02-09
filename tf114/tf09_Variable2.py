import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
x = [1,2,3]
y = [3,5,7]
w = tf.Variable([0.3])
b = tf.Variable([1.0])

x_train = tf.placeholder(tf.float32, shape=[None])  
y_train = tf.placeholder(tf.float32, shape=[None])  
x_test = tf.placeholder(tf.float32, shape=[None])  

#2. 모델구성
hypothesis = x * w + b

############################################### 1. Session() // sess.run   #####################################################

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                        feed_dict={x_train:x, y_train:y})
    # if step % 20 == 0:
    #     # print(step, sess.run(loss), sess.run(w), sess.run(b))
    #     print(step, loss_val, w_val, b_val)
    
#4. 예측
predict = x_test * w_val + b_val      # predict = model.predict

print("1. [6,7,8] 예측 : " , sess.run(predict, feed_dict={x_test:[6,7,8]}))

sess.close()

############################################### 2. Session() // eval #####################################################

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                        feed_dict={x_train:x, y_train:y})
    # if step % 20 == 0:
    #     # print(step, sess.run(loss), sess.run(w), sess.run(b))
    #     print(step, loss_val, w_val, b_val)
    
#4. 예측
predict = x_test * w_val + b_val      # predict = model.predict

print("2. [6,7,8] 예측 : " , predict.eval(session=sess, feed_dict={x_test:[6,7,8]}))

sess.close()

############################################### 3. InteractiveSession() // eval #####################################################

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                        feed_dict={x_train:x, y_train:y})
    # if step % 20 == 0:
    #     # print(step, sess.run(loss), sess.run(w), sess.run(b))
    #     print(step, loss_val, w_val, b_val)
    
#4. 예측
predict = x_test * w_val + b_val      # predict = model.predict

print("3. [6,7,8] 예측 : " , predict.eval(feed_dict={x_test:[6,7,8]}))

sess.close()


